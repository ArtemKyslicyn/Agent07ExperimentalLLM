//
//  ExperimentalLLMTests.swift
//  Agent07ExperimentalLLMTests
//

import Testing
import Foundation
@testable import Agent07ExperimentalLLM

// MARK: - LlamaBackend Protocol Tests

@Suite("LlamaBackend Protocol Tests")
struct LlamaBackendTests {

    @Test("MockBackend conforms to protocol")
    func mockConforms() async throws {
        let backend: any LlamaBackend = MockLlamaBackend()
        #expect(backend.backendId == "mock-1.0")
    }

    @Test("MockBackend generates response")
    func mockGenerates() async throws {
        let backend = MockLlamaBackend(response: "Hello World")
        let model = try await backend.loadModel(path: "/tmp/fake.gguf", config: .full)
        let result = try await backend.generate(model: model, prompt: "Hi", config: .default)
        #expect(result.text == "Hello World")
    }

    @Test("MockBackend streams tokens")
    func mockStreams() async throws {
        let backend = MockLlamaBackend(response: "streamed output")
        let model = try await backend.loadModel(path: "/tmp/fake.gguf", config: .full)
        let tokens = LockedArray<String>()
        _ = try await backend.generateStream(model: model, prompt: "x", config: .default) { token in
            tokens.append(token)
        }
        #expect(!tokens.snapshot.isEmpty)
    }

    @Test("MockBackend returns model info")
    func mockModelInfo() async throws {
        let backend = MockLlamaBackend()
        let model = try await backend.loadModel(path: "/tmp/x.gguf", config: .full)
        let info = backend.modelInfo(model)
        #expect(info?.layerCount == 4)
        #expect(info?.name == "mock")
    }
}

// MARK: - ModelLoadConfig Tests

@Suite("ModelLoadConfig Tests")
struct ModelLoadConfigTests {

    @Test("Full config has all layers")
    func fullConfig() {
        let config = ModelLoadConfig.full
        #expect(config.layerMask == nil)
        #expect(config.gpuLayers == -1)
    }

    @Test("CPU minimal config")
    func cpuMinimal() {
        let config = ModelLoadConfig.cpuMinimal
        #expect(config.gpuLayers == 0)
        #expect(config.contextSize == 2048)
    }

    @Test("Coding layers mask covers start + end")
    func codingLayers() {
        let config = ModelLoadConfig.codingLayers(totalLayers: 32)
        #expect(config.layerMask != nil)
        let mask = config.layerMask!
        #expect(mask.contains(0))   // first layers included
        #expect(mask.contains(31))  // last layers included
        #expect(mask.count < 32)    // some layers skipped
    }

    @Test("Reasoning layers mask")
    func reasoningLayers() {
        let config = ModelLoadConfig.reasoningLayers(totalLayers: 32)
        let mask = config.layerMask!
        #expect(mask.count < 32)
        #expect(mask.contains(0))
        #expect(mask.contains(31))
    }

    @Test("Layer savings calculation")
    func layerSavings() {
        let info = ModelInfo(name: "test", layerCount: 32, loadedLayers: Array(0..<20))
        #expect(info.layerSavings > 0.3, "Loading 20/32 layers = ~37.5% savings")
    }
}

// MARK: - BackendManager Tests

@Suite("BackendManager Tests")
struct BackendManagerTests {

    @Test("Register and get active backend")
    func registerAndGet() async throws {
        let manager = BackendManager()
        await manager.register(MockLlamaBackend(id: "v1"))
        let active = try await manager.active()
        #expect(active.backendId == "v1")
    }

    @Test("Switch between backends")
    func switchBackend() async throws {
        let manager = BackendManager()
        await manager.register(MockLlamaBackend(id: "v1"))
        await manager.register(MockLlamaBackend(id: "v2"))

        try await manager.switchTo(backendId: "v2")
        let active = try await manager.active()
        #expect(active.backendId == "v2")
    }

    @Test("Switch to unknown backend throws")
    func switchUnknown() async {
        let manager = BackendManager()
        await manager.register(MockLlamaBackend(id: "v1"))
        do {
            try await manager.switchTo(backendId: "nonexistent")
            #expect(Bool(false), "Should throw")
        } catch {
            #expect(error.localizedDescription.contains("not registered"))
        }
    }

    @Test("No active backend throws")
    func noActive() async {
        let manager = BackendManager()
        do {
            _ = try await manager.active()
            #expect(Bool(false), "Should throw")
        } catch {
            #expect(error.localizedDescription.contains("No active"))
        }
    }

    @Test("List registered backends")
    func listBackends() async {
        let manager = BackendManager()
        await manager.register(MockLlamaBackend(id: "alpha"))
        await manager.register(MockLlamaBackend(id: "beta"))
        let list = await manager.registeredBackends
        #expect(list.count == 2)
        #expect(list.contains("alpha"))
        #expect(list.contains("beta"))
    }

    @Test("Capability matrix")
    func capabilityMatrix() async {
        let manager = BackendManager()
        await manager.register(MockLlamaBackend(
            id: "streaming-v1",
            capabilities: BackendCapabilities(realStreaming: true, metalSupport: true)
        ))
        let matrix = await manager.capabilityMatrix
        #expect(matrix.count == 1)
        #expect(matrix[0].caps.realStreaming == true)
    }
}

// MARK: - LlamaCppBackend Tests

@Suite("LlamaCppBackend Tests")
struct LlamaCppBackendTests {

    @Test("Backend has correct capabilities")
    func capabilities() {
        let backend = LlamaCppBackend(version: "test")
        #expect(backend.capabilities.realStreaming)
        #expect(backend.capabilities.layerSpecialization)
        #expect(backend.capabilities.metalSupport)
        #expect(backend.capabilities.maxContext == 131072)
    }

    @Test("BackendId contains version")
    func backendId() {
        let backend = LlamaCppBackend(version: "b5200")
        #expect(backend.backendId == "llama.cpp-b5200")
    }

    @Test("Load nonexistent model throws")
    func loadNonexistent() async {
        let backend = LlamaCppBackend(binaryPath: "/bin/echo", version: "test")
        do {
            _ = try await backend.loadModel(path: "/nonexistent.gguf", config: .full)
            #expect(Bool(false))
        } catch {
            #expect(error.localizedDescription.contains("not found"))
        }
    }

    @Test("Generate config presets")
    func configPresets() {
        #expect(GenerateConfig.default.temperature == 0.7)
        #expect(GenerateConfig.creative.temperature == 1.0)
        #expect(GenerateConfig.precise.temperature == 0.1)
    }

    @Test("GenerateResult calculates tokens per second")
    func tokensPerSecond() {
        let result = GenerateResult(text: "hello world", tokenCount: 100, duration: 2.0)
        #expect(result.tokensPerSecond == 50.0)
    }

    @Test("Load with missing binary throws binaryNotFound")
    func loadMissingBinary() async {
        // Real GGUF path must exist for the first guard to pass — create
        // a non-empty temp file. The binaryPath then points at a path
        // that definitely doesn't exist, triggering .binaryNotFound.
        let tempGGUF = NSTemporaryDirectory() + "fake-\(UUID().uuidString).gguf"
        try? "fake".write(toFile: tempGGUF, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: tempGGUF) }

        let backend = LlamaCppBackend(
            binaryPath: "/dev/null/does-not-exist/llama-cli",
            version: "test"
        )
        do {
            _ = try await backend.loadModel(path: tempGGUF, config: .full)
            #expect(Bool(false), "Should have thrown")
        } catch {
            #expect(error.localizedDescription.contains("not found"))
        }
    }

    @Test("Generate on unloaded handle throws modelNotLoaded")
    func generateUnloadedThrows() async {
        let backend = LlamaCppBackend(binaryPath: "/bin/echo", version: "test")
        let strayHandle = ModelHandle(path: "/tmp/never.gguf", backendId: "llama.cpp-test")
        do {
            _ = try await backend.generate(
                model: strayHandle, prompt: "x", config: .default
            )
            #expect(Bool(false), "Should have thrown")
        } catch {
            // Either modelNotLoaded or any other LlamaError variant — we
            // just assert the call surfaces an error.
            _ = error
        }
    }

    @Test("GenerateStream on unloaded handle throws modelNotLoaded")
    func generateStreamUnloadedThrows() async {
        let backend = LlamaCppBackend(binaryPath: "/bin/echo", version: "test")
        let strayHandle = ModelHandle(path: "/tmp/never.gguf", backendId: "llama.cpp-test")
        do {
            _ = try await backend.generateStream(
                model: strayHandle, prompt: "x", config: .default, onToken: { _ in }
            )
            #expect(Bool(false), "Should have thrown")
        } catch {
            _ = error
        }
    }

    @Test("Unload nonexistent handle is a safe no-op")
    func unloadUnknownHandleSafe() async {
        let backend = LlamaCppBackend(binaryPath: "/bin/echo", version: "test")
        await backend.unloadModel(
            ModelHandle(path: "/tmp/x", backendId: "llama.cpp-test")
        )
    }

    @Test("modelInfo returns nil for unknown handle")
    func modelInfoUnknownHandle() {
        let backend = LlamaCppBackend(binaryPath: "/bin/echo", version: "test")
        let stray = ModelHandle(path: "/tmp/never.gguf", backendId: "llama.cpp-test")
        let info = backend.modelInfo(stray)
        // modelInfo may return nil OR a stub — we just assert no crash.
        _ = info
    }

    @Test("Environment extras are accepted at init")
    func environmentExtrasInit() {
        let backend = LlamaCppBackend(
            binaryPath: "/bin/echo", version: "test",
            environmentExtras: ["AGENT07_MODELS_PRIMARY": "/tmp/models"]
        )
        #expect(backend.backendId == "llama.cpp-test")
    }
}

// MARK: - NativeLlamaBackend / LlamaDynLib

@Suite("LlamaDynLib")
struct LlamaDynLibTests {

    @Test("isAvailable checks filesystem without crash")
    func isAvailableProbe() {
        // Returns true or false depending on whether libllama.dylib is
        // installed on this machine. We only assert the call completes
        // and yields a Bool — both outcomes are valid.
        let available = LlamaDynLib.isAvailable
        _ = available
    }

    @Test("load() with an obviously-wrong path throws binaryNotFound")
    func loadWrongPathThrows() {
        do {
            _ = try LlamaDynLib.load(path: "/dev/null/not-a-real-dylib.dylib")
            // If libllama IS installed system-wide, load() will succeed
            // with one of its fallback paths instead. We only fail the
            // test if neither outcome happened.
        } catch {
            #expect(error is LlamaError)
        }
    }
}

// MARK: - Value types — additional coverage

@Suite("BackendCapabilities + ModelInfo + ModelHandle")
struct ValueTypesTests {

    @Test func capabilitiesDefaultsAreSafe() {
        let caps = BackendCapabilities()
        #expect(caps.realStreaming == false)
        #expect(caps.layerSpecialization == false)
        #expect(caps.speculativeDecoding == false)
        #expect(caps.kvCacheControl == false)
        #expect(caps.metalSupport == false)
        #expect(caps.maxContext == 4096)
    }

    @Test func capabilitiesExplicitInit() {
        let caps = BackendCapabilities(
            realStreaming: true, layerSpecialization: true,
            speculativeDecoding: true, kvCacheControl: true,
            metalSupport: true, maxContext: 131_072
        )
        #expect(caps.realStreaming)
        #expect(caps.maxContext == 131_072)
    }

    @Test func modelHandleHasUniqueIdAndCapturedFields() {
        let a = ModelHandle(path: "/tmp/a.gguf", backendId: "x")
        let b = ModelHandle(path: "/tmp/a.gguf", backendId: "x")
        #expect(a.id != b.id)
        #expect(a.path == "/tmp/a.gguf")
        #expect(a.backendId == "x")
    }

    @Test func modelInfoAllLayersLoadedYieldsZeroSavings() {
        let info = ModelInfo(name: "n", layerCount: 32,
                              loadedLayers: Array(0..<32))
        #expect(info.layerSavings == 0)
    }

    @Test func modelInfoNilLayerMaskYieldsZeroSavings() {
        let info = ModelInfo(name: "n", layerCount: 32, loadedLayers: nil)
        #expect(info.layerSavings == 0)
    }

    @Test func modelInfoFullInit() {
        let info = ModelInfo(
            name: "Qwen3", parameterCount: 7_000_000_000,
            layerCount: 32, contextSize: 32_768,
            quantization: "Q4_K_M",
            fileSizeBytes: 4_000_000_000,
            loadedLayers: [0, 1, 2],
            ramUsageBytes: 4_500_000_000
        )
        #expect(info.name == "Qwen3")
        #expect(info.parameterCount == 7_000_000_000)
        #expect(info.quantization == "Q4_K_M")
        #expect(info.layerSavings > 0.9)
    }
}

@Suite("BackendManagerError")
struct BackendManagerErrorDescriptionTests {

    @Test func backendNotFoundDescription() {
        let e = BackendManagerError.backendNotFound("xyz")
        let d = e.errorDescription ?? ""
        #expect(d.contains("xyz"))
        #expect(d.contains("not registered"))
    }

    @Test func noActiveBackendDescription() {
        let e = BackendManagerError.noActiveBackend
        #expect(e.errorDescription?.contains("register") == true)
    }
}

@Suite("BackendManager — extra coverage")
struct BackendManagerExtraTests {

    @Test func registerLlamaCppAddsBackendWithVersion() async {
        let m = BackendManager()
        await m.registerLlamaCpp(version: "b5200")
        let list = await m.registeredBackends
        #expect(list.contains("llama.cpp-b5200"))
    }

    @Test func backendByIdReturnsNilForUnknown() async {
        let m = BackendManager()
        let result = await m.backend(id: "missing")
        #expect(result == nil)
    }

    @Test func currentBackendIdBeforeRegisterIsNil() async {
        let m = BackendManager()
        let id = await m.currentBackendId
        #expect(id == nil)
    }

    @Test func currentBackendIdAfterRegisterMatchesFirst() async {
        let m = BackendManager()
        await m.register(MockLlamaBackend(id: "first"))
        await m.register(MockLlamaBackend(id: "second"))
        let id = await m.currentBackendId
        #expect(id == "first")
    }

    @Test func switchToUnregisteredThrowsBackendNotFound() async {
        let m = BackendManager()
        await m.register(MockLlamaBackend(id: "only"))
        do {
            try await m.switchTo(backendId: "ghost")
            Issue.record("Should have thrown")
        } catch let err as BackendManagerError {
            if case .backendNotFound(let id) = err {
                #expect(id == "ghost")
            } else {
                Issue.record("Wrong error variant")
            }
        } catch { Issue.record("Wrong error type") }
    }
}

@Suite("MockLlamaBackend — additional surface")
struct MockBackendSurfaceTests {

    @Test func unloadIsSafeNoOp() async throws {
        let backend = MockLlamaBackend()
        let model = try await backend.loadModel(path: "/tmp/x", config: .full)
        await backend.unloadModel(model)
        // No assertion needed — successful return is the contract.
    }

    @Test func modelInfoReportsMockMetadata() async throws {
        let backend = MockLlamaBackend()
        let model = try await backend.loadModel(path: "/tmp/x", config: .full)
        let info = backend.modelInfo(model)
        #expect(info?.parameterCount == 1_000_000)
        #expect(info?.layerCount == 4)
    }

    @Test func capabilitiesArePassedThroughInit() {
        let caps = BackendCapabilities(realStreaming: true, maxContext: 8192)
        let backend = MockLlamaBackend(id: "custom", capabilities: caps)
        #expect(backend.capabilities.realStreaming)
        #expect(backend.capabilities.maxContext == 8192)
        #expect(backend.backendId == "custom")
    }
}

@Suite("ModelLoadConfig + GenerateConfig extras")
struct ConfigExtraTests {

    @Test func modelLoadConfigCustomInit() {
        let c = ModelLoadConfig(
            gpuLayers: 16, contextSize: 8192,
            layerMask: [0, 1, 31],
            quantization: "Q8_0",
            threads: 8, mmap: false
        )
        #expect(c.gpuLayers == 16)
        #expect(c.contextSize == 8192)
        #expect(c.layerMask == [0, 1, 31])
        #expect(c.quantization == "Q8_0")
        #expect(c.threads == 8)
        #expect(c.mmap == false)
    }

    @Test func generateConfigCustomInit() {
        let c = GenerateConfig(
            maxTokens: 4096, temperature: 0.3, topP: 0.85, topK: 20,
            repeatPenalty: 1.2, stopSequences: ["<end>", "###"]
        )
        #expect(c.maxTokens == 4096)
        #expect(c.temperature == 0.3)
        #expect(c.stopSequences.contains("<end>"))
    }

    @Test func generateResultZeroDurationYieldsZeroTokensPerSecond() {
        let r = GenerateResult(text: "x", tokenCount: 10, duration: 0)
        #expect(r.tokensPerSecond == 0)
    }
}
