//
//  BackendManager.swift
//  Agent07ExperimentalLLM
//
//  Manages multiple LLM backends. Allows switching between versions at runtime.
//  Production: LLM.swift (via Agent07LLMServices, not touched here)
//  Experimental: LlamaCppBackend (this package)
//  Future: MLXBackend, CoreMLBackend
//

import Foundation

// MARK: - Backend Registry

public actor BackendManager {

    /// All registered backends
    private var backends: [String: any LlamaBackend] = [:]

    /// Currently active backend ID
    private var activeBackendId: String?

    public init() {}

    // MARK: - Registration

    /// Register a backend.
    public func register(_ backend: any LlamaBackend) {
        backends[backend.backendId] = backend
        if activeBackendId == nil {
            activeBackendId = backend.backendId
        }
    }

    /// Register the default llama.cpp backend with auto-detected binary.
    public func registerLlamaCpp(version: String = "latest", binaryPath: String? = nil) {
        let backend = LlamaCppBackend(binaryPath: binaryPath, version: version)
        register(backend)
    }

    // MARK: - Switching

    /// Switch to a different backend.
    public func switchTo(backendId: String) throws {
        guard backends[backendId] != nil else {
            throw BackendManagerError.backendNotFound(backendId)
        }
        activeBackendId = backendId
    }

    /// Get the currently active backend.
    public func active() throws -> any LlamaBackend {
        guard let id = activeBackendId, let backend = backends[id] else {
            throw BackendManagerError.noActiveBackend
        }
        return backend
    }

    // MARK: - Query

    /// List all registered backend IDs.
    public var registeredBackends: [String] {
        Array(backends.keys).sorted()
    }

    /// Get backend by ID.
    public func backend(id: String) -> (any LlamaBackend)? {
        backends[id]
    }

    /// Current active backend ID.
    public var currentBackendId: String? { activeBackendId }

    /// Compare capabilities of all backends.
    public var capabilityMatrix: [(id: String, caps: BackendCapabilities)] {
        backends.map { ($0.key, $0.value.capabilities) }
    }
}

// MARK: - Errors

public enum BackendManagerError: LocalizedError, Sendable {
    case backendNotFound(String)
    case noActiveBackend

    public var errorDescription: String? {
        switch self {
        case .backendNotFound(let id): return "Backend '\(id)' not registered"
        case .noActiveBackend: return "No active backend. Call register() first."
        }
    }
}

// MARK: - Mock Backend (for testing)

public final class MockLlamaBackend: LlamaBackend, @unchecked Sendable {
    public let backendId: String
    public let capabilities: BackendCapabilities
    private let mockResponse: String

    public init(id: String = "mock-1.0", response: String = "mock output",
                capabilities: BackendCapabilities = .init()) {
        self.backendId = id; self.mockResponse = response; self.capabilities = capabilities
    }

    public func loadModel(path: String, config: ModelLoadConfig) async throws -> ModelHandle {
        ModelHandle(path: path, backendId: backendId)
    }

    public func generate(model: ModelHandle, prompt: String, config: GenerateConfig) async throws -> GenerateResult {
        GenerateResult(text: mockResponse, tokenCount: mockResponse.count / 4, duration: 0.01)
    }

    public func generateStream(model: ModelHandle, prompt: String, config: GenerateConfig,
                                onToken: @escaping @Sendable (String) -> Void) async throws -> GenerateResult {
        onToken(mockResponse)
        return GenerateResult(text: mockResponse, tokenCount: mockResponse.count / 4, duration: 0.01)
    }

    public func unloadModel(_ handle: ModelHandle) async {}

    public func modelInfo(_ handle: ModelHandle) -> ModelInfo? {
        ModelInfo(name: "mock", parameterCount: 1_000_000, layerCount: 4)
    }
}
