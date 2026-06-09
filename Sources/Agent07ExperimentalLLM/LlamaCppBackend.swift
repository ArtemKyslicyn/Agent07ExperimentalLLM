//
//  LlamaCppBackend.swift
//  Agent07ExperimentalLLM
//
//  Direct llama.cpp integration via CLI subprocess.
//  Uses llama-cli (installed via `brew install llama.cpp` or built from source).
//  This is the EXPERIMENTAL backend — production uses LLM.swift.
//
//  Version switching: change `binaryPath` to point to different llama.cpp builds.
//

import Foundation
import os.log

// MARK: - Llama.cpp Backend

private let llamaCppLog = Logger(subsystem: "Agent07ExperimentalLLM", category: "LlamaCppBackend")

public final class LlamaCppBackend: LlamaBackend, @unchecked Sendable {

    public let backendId: String
    private let binaryPath: String     // path to llama-cli or llama-server
    /// Merged into each `Process.environment` (e.g. Agent07 model dirs + active GGUF path).
    private let environmentExtras: [String: String]
    private var loadedModels: [UUID: ModelHandle] = [:]
    private var modelPaths: [UUID: String] = [:]
    /// Per-handle load config so generation uses the REAL context window the caller
    /// requested at loadModel time (was previously dropped → generate hardcoded .full → 4096).
    private var loadConfigs: [UUID: ModelLoadConfig] = [:]

    public var capabilities: BackendCapabilities {
        BackendCapabilities(
            realStreaming: true,         // llama.cpp supports real streaming
            layerSpecialization: true,   // --n-gpu-layers controls offloading
            speculativeDecoding: true,   // --draft-model support
            kvCacheControl: true,        // --cache-type-k, --cache-type-v
            metalSupport: true,          // --n-gpu-layers -1 for full Metal
            maxContext: 131072           // llama.cpp supports 128K+
        )
    }

    /// Initialize with specific llama.cpp version.
    /// - binaryPath: path to llama-cli binary (e.g. /opt/homebrew/bin/llama-cli)
    /// - version: version tag for identification (e.g. "b5200")
    /// - environmentExtras: merged into subprocess env (e.g. `AGENT07_MODELS_PRIMARY`).
    public init(binaryPath: String? = nil, version: String = "latest", environmentExtras: [String: String] = [:]) {
        self.binaryPath = binaryPath ?? Self.findBinary()
        self.backendId = "llama.cpp-\(version)"
        self.environmentExtras = environmentExtras
    }

    private func processEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        for (key, value) in environmentExtras {
            env[key] = value
        }
        return env
    }

    // MARK: - Model Loading

    public func loadModel(path: String, config: ModelLoadConfig) async throws -> ModelHandle {
        guard FileManager.default.fileExists(atPath: path) else {
            llamaCppLog.error("loadModel: GGUF not found at \(path, privacy: .public)")
            throw LlamaError.modelNotFound(path)
        }
        guard FileManager.default.fileExists(atPath: binaryPath) else {
            llamaCppLog.error("loadModel: llama binary not found at \(self.binaryPath, privacy: .public) — brew install llama.cpp")
            throw LlamaError.binaryNotFound(binaryPath)
        }

        let handle = ModelHandle(path: path, backendId: backendId)
        loadedModels[handle.id] = handle
        modelPaths[handle.id] = path
        loadConfigs[handle.id] = config
        return handle
    }

    public func unloadModel(_ handle: ModelHandle) async {
        loadedModels.removeValue(forKey: handle.id)
        modelPaths.removeValue(forKey: handle.id)
        loadConfigs.removeValue(forKey: handle.id)
    }

    // MARK: - Generation

    public func generate(model: ModelHandle, prompt: String, config: GenerateConfig) async throws -> GenerateResult {
        guard let modelPath = modelPaths[model.id] else {
            throw LlamaError.modelNotLoaded
        }

        let start = Date()
        let output = try await runLlamaCli(
            modelPath: modelPath,
            prompt: prompt,
            config: config,
            loadConfig: loadConfigs[model.id] ?? .full
        )
        let duration = Date().timeIntervalSince(start)
        let tokenEstimate = output.components(separatedBy: .whitespacesAndNewlines).count

        return GenerateResult(text: output, tokenCount: tokenEstimate, duration: duration)
    }

    public func generateStream(model: ModelHandle, prompt: String, config: GenerateConfig,
                                onToken: @escaping @Sendable (String) -> Void) async throws -> GenerateResult {
        guard let modelPath = modelPaths[model.id] else {
            throw LlamaError.modelNotLoaded
        }

        let start = Date()
        // Use the same path as `generate` (blocking read of stdout). Pipe + readabilityHandler
        // is unreliable in GUI apps — often produced 0 chars while the CLI worked.
        let output = try await runLlamaCli(
            modelPath: modelPath,
            prompt: prompt,
            config: config,
            loadConfig: loadConfigs[model.id] ?? .full
        )
        let duration = Date().timeIntervalSince(start)
        if !output.isEmpty {
            let chunkSize = max(1, output.count / 24)
            var idx = output.startIndex
            while idx < output.endIndex {
                let end = output.index(idx, offsetBy: chunkSize, limitedBy: output.endIndex) ?? output.endIndex
                onToken(String(output[idx..<end]))
                idx = end
            }
        }
        let tokenEstimate = output.components(separatedBy: .whitespacesAndNewlines).count

        return GenerateResult(text: output, tokenCount: tokenEstimate, duration: duration)
    }

    public func modelInfo(_ handle: ModelHandle) -> ModelInfo? {
        guard let path = modelPaths[handle.id] else { return nil }
        let name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? UInt64) ?? 0
        return ModelInfo(name: name, fileSizeBytes: size)
    }

    // MARK: - CLI Execution

    private func runLlamaCli(modelPath: String, prompt: String, config: GenerateConfig,
                              loadConfig: ModelLoadConfig) async throws -> String {
        var args: [String] = [
            "-m", modelPath,
            "-p", prompt,
            "-n", String(config.maxTokens),
            "--temp", String(config.temperature),
            "--top-p", String(config.topP),
            "--top-k", String(config.topK),
            "--repeat-penalty", String(config.repeatPenalty),
            "-t", String(loadConfig.threads),
            "--no-display-prompt",
            // Homebrew llama.cpp b8680+: default is conversation mode — would block or yield no stdout for -p
            "-no-cnv"
        ]

        if loadConfig.gpuLayers >= 0 {
            args += ["-ngl", String(loadConfig.gpuLayers)]
        } else {
            args += ["-ngl", "99"] // offload all layers
        }

        if loadConfig.contextSize > 0 {
            args += ["-c", String(loadConfig.contextSize)]
        }

        // Layer specialization: use --n-gpu-layers with specific count
        if let mask = loadConfig.layerMask {
            // llama.cpp doesn't support arbitrary layer masks natively,
            // but we can approximate by setting n-gpu-layers to the masked count
            args += ["--n-gpu-layers", String(mask.count)]
        }

        // Fix #1: scale the subprocess timeout with the generation budget. The old fixed
        // 60s killed big/reasoning models mid-turn (~18 tok/s → only ~1080 tokens fit 60s).
        // Budget ≈ load margin (120s) + maxTokens at a conservative 8 tok/s floor, capped
        // at 30 min so a runaway can't hang forever.
        let scaledTimeout = min(1800, 120 + config.maxTokens / 8)

        // Fix #5: `--stop` is rejected by Homebrew b8680+, so approximate stop sequences by
        // POST-TRUNCATING the output at the earliest stop marker (prevents runaway past the
        // assistant turn into fake <|im_start|> continuations). Model EOS still ends most runs.
        var output = try runProcess(binaryPath, arguments: args, timeoutSeconds: scaledTimeout)
        var cut = output.endIndex
        for stop in config.stopSequences {
            if let r = output.range(of: stop), r.lowerBound < cut { cut = r.lowerBound }
        }
        if cut < output.endIndex { output = String(output[..<cut]) }
        return output
    }

    private func runProcess(_ path: String, arguments: [String], timeoutSeconds: Int = 60) throws -> String {
        let process = Process()
        let pipe = Pipe()
        let errPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: path)
        process.environment = processEnvironment()
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = errPipe
        try process.run()

        // Timeout: kill process if it takes too long
        let deadline = DispatchTime.now() + .seconds(timeoutSeconds)
        DispatchQueue.global().asyncAfter(deadline: deadline) {
            if process.isRunning { process.terminate() }
        }
        process.waitUntilExit()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let errText = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        if output.isEmpty {
            llamaCppLog.error("runProcess: empty stdout exit=\(process.terminationStatus) stderr=\(errText.prefix(2000), privacy: .public)")
        } else if !errText.isEmpty {
            llamaCppLog.warning("runProcess: stderr (non-fatal)=\(errText.prefix(1200), privacy: .public)")
        }

        // Process may have been killed by timeout — still return partial output
        if process.terminationStatus != 0 && output.isEmpty {
            throw LlamaError.processError(exitCode: Int(process.terminationStatus), stderr: errText)
        }
        return output
    }

    // MARK: - Binary Discovery

    private static func findBinary() -> String {
        // Prefer llama-completion (non-interactive, single-shot)
        let candidates = [
            "/opt/homebrew/bin/llama-completion",
            "/usr/local/bin/llama-completion",
            "/opt/homebrew/bin/llama-cli",
            "/usr/local/bin/llama-cli",
            NSHomeDirectory() + "/.local/bin/llama-completion",
            NSHomeDirectory() + "/llama.cpp/build/bin/llama-completion"
        ]
        return candidates.first(where: { FileManager.default.fileExists(atPath: $0) })
            ?? "/opt/homebrew/bin/llama-completion"
    }
}

// MARK: - Errors

public enum LlamaError: LocalizedError, Sendable {
    case modelNotFound(String)
    case binaryNotFound(String)
    case modelNotLoaded
    case processError(exitCode: Int, stderr: String)
    case layerMaskNotSupported

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let p): return "Model not found: \(p)"
        case .binaryNotFound(let p): return "llama-cli not found at \(p). Install: brew install llama.cpp"
        case .modelNotLoaded: return "Model not loaded"
        case .processError(let code, let stderr):
            if stderr.isEmpty {
                return "llama-cli exited with code \(code)"
            }
            return "llama-cli exited with code \(code): \(stderr)"
        case .layerMaskNotSupported: return "Layer mask not supported by this backend version"
        }
    }
}
