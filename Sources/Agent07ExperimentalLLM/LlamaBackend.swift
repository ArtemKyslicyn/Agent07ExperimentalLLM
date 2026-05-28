//
//  LlamaBackend.swift
//  Agent07ExperimentalLLM
//
//  Protocol for switching between llama.cpp versions.
//  Production: LLMSwiftBackend (wraps LLM.swift v2.1.0)
//  Experimental: LlamaCppBackend (direct llama.cpp via CLI or C bridge)
//  Future: MLXBackend, CoreMLBackend
//

import Foundation

// MARK: - Backend Protocol

/// Universal LLM backend protocol. Implementations can use any inference engine.
public protocol LlamaBackend: Sendable {
    /// Backend identifier (e.g. "llm.swift-2.1.0", "llama.cpp-b5200", "mlx-0.3")
    var backendId: String { get }

    /// Supported features
    var capabilities: BackendCapabilities { get }

    /// Load model from file path. Returns model handle.
    func loadModel(path: String, config: ModelLoadConfig) async throws -> ModelHandle

    /// Generate completion from prompt.
    func generate(model: ModelHandle, prompt: String, config: GenerateConfig) async throws -> GenerateResult

    /// Generate with token-by-token streaming.
    func generateStream(model: ModelHandle, prompt: String, config: GenerateConfig,
                        onToken: @escaping @Sendable (String) -> Void) async throws -> GenerateResult

    /// Unload model from memory.
    func unloadModel(_ handle: ModelHandle) async

    /// Get info about loaded model (layers, parameters, etc.)
    func modelInfo(_ handle: ModelHandle) -> ModelInfo?
}

// MARK: - Capabilities

public struct BackendCapabilities: Sendable {
    /// Supports real token-by-token streaming (not chunked)
    public let realStreaming: Bool

    /// Supports layer-level control (load/skip specific layers)
    public let layerSpecialization: Bool

    /// Supports speculative decoding natively
    public let speculativeDecoding: Bool

    /// Supports KV cache management
    public let kvCacheControl: Bool

    /// Supports GPU/Metal offloading
    public let metalSupport: Bool

    /// Max context size supported
    public let maxContext: Int

    public init(realStreaming: Bool = false, layerSpecialization: Bool = false,
                speculativeDecoding: Bool = false, kvCacheControl: Bool = false,
                metalSupport: Bool = false, maxContext: Int = 4096) {
        self.realStreaming = realStreaming; self.layerSpecialization = layerSpecialization
        self.speculativeDecoding = speculativeDecoding; self.kvCacheControl = kvCacheControl
        self.metalSupport = metalSupport; self.maxContext = maxContext
    }
}

// MARK: - Model Handle

public struct ModelHandle: Identifiable, Sendable {
    public let id: UUID
    public let path: String
    public let backendId: String
    public let loadedAt: Date

    public init(path: String, backendId: String) {
        self.id = UUID(); self.path = path
        self.backendId = backendId; self.loadedAt = Date()
    }
}

// MARK: - Model Load Config

public struct ModelLoadConfig: Sendable {
    /// Number of GPU layers to offload (-1 = all, 0 = CPU only)
    public var gpuLayers: Int

    /// Context size in tokens
    public var contextSize: Int

    /// Layer mask: which transformer layers to load (nil = all)
    /// For layer specialization: [0,1,2,3,4, 15,16,17,...,31] = skip middle layers
    public var layerMask: [Int]?

    /// Quantization type to use when loading
    public var quantization: String?

    /// Threading
    public var threads: Int

    /// Memory-map the model (faster load, uses less RAM initially)
    public var mmap: Bool

    public init(gpuLayers: Int = -1, contextSize: Int = 4096, layerMask: [Int]? = nil,
                quantization: String? = nil, threads: Int = 4, mmap: Bool = true) {
        self.gpuLayers = gpuLayers; self.contextSize = contextSize
        self.layerMask = layerMask; self.quantization = quantization
        self.threads = threads; self.mmap = mmap
    }

    // MARK: - Preset Configs

    /// Full model, all layers, GPU accelerated
    public static let full = ModelLoadConfig()

    /// CPU-only, minimal context (for tiny models)
    public static let cpuMinimal = ModelLoadConfig(gpuLayers: 0, contextSize: 2048, threads: 2)

    /// Layer-specialized: only coding-relevant layers
    public static func codingLayers(totalLayers: Int) -> ModelLoadConfig {
        let mask = Array(0..<min(20, totalLayers)) + Array(max(0, totalLayers - 5)..<totalLayers)
        return ModelLoadConfig(layerMask: mask)
    }

    /// Layer-specialized: only reasoning layers
    public static func reasoningLayers(totalLayers: Int) -> ModelLoadConfig {
        let mask = Array(0..<min(15, totalLayers)) + Array(max(0, totalLayers - 10)..<totalLayers)
        return ModelLoadConfig(layerMask: mask)
    }
}

// MARK: - Generate Config

public struct GenerateConfig: Sendable {
    public var maxTokens: Int
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var repeatPenalty: Double
    public var stopSequences: [String]

    public init(maxTokens: Int = 2048, temperature: Double = 0.7, topP: Double = 0.9,
                topK: Int = 40, repeatPenalty: Double = 1.1, stopSequences: [String] = []) {
        self.maxTokens = maxTokens; self.temperature = temperature; self.topP = topP
        self.topK = topK; self.repeatPenalty = repeatPenalty; self.stopSequences = stopSequences
    }

    public static let `default` = GenerateConfig()
    public static let creative = GenerateConfig(temperature: 1.0, topP: 0.95)
    public static let precise = GenerateConfig(temperature: 0.1, topP: 0.5, topK: 10)
}

// MARK: - Generate Result

public struct GenerateResult: Sendable {
    public let text: String
    public let tokenCount: Int
    public let duration: TimeInterval
    public let tokensPerSecond: Double

    public init(text: String, tokenCount: Int, duration: TimeInterval) {
        self.text = text; self.tokenCount = tokenCount; self.duration = duration
        self.tokensPerSecond = duration > 0 ? Double(tokenCount) / duration : 0
    }
}

// MARK: - Model Info

public struct ModelInfo: Sendable {
    public let name: String
    public let parameterCount: Int64     // e.g. 7_000_000_000
    public let layerCount: Int           // e.g. 32
    public let contextSize: Int          // e.g. 4096
    public let quantization: String      // e.g. "Q4_K_M"
    public let fileSizeBytes: UInt64
    public let loadedLayers: [Int]?      // nil = all, [0,1,2,...] = subset
    public let ramUsageBytes: UInt64     // estimated

    public init(name: String, parameterCount: Int64 = 0, layerCount: Int = 32,
                contextSize: Int = 4096, quantization: String = "Q4_K_M",
                fileSizeBytes: UInt64 = 0, loadedLayers: [Int]? = nil, ramUsageBytes: UInt64 = 0) {
        self.name = name; self.parameterCount = parameterCount
        self.layerCount = layerCount; self.contextSize = contextSize
        self.quantization = quantization; self.fileSizeBytes = fileSizeBytes
        self.loadedLayers = loadedLayers; self.ramUsageBytes = ramUsageBytes
    }

    /// RAM savings from layer specialization (0.0-1.0)
    public var layerSavings: Double {
        guard let loaded = loadedLayers, layerCount > 0 else { return 0 }
        return 1.0 - Double(loaded.count) / Double(layerCount)
    }
}
