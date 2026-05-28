# Agent07ExperimentalLLM

A pluggable, runtime-loadable LLM backend layer for Swift. Switch between
llama.cpp builds, mock backends, and (soon) MLX without recompiling — and
without symbol collisions when multiple llama.cpp variants live in the
same process.

[![Swift 6](https://img.shields.io/badge/Swift-6.0-orange)](https://swift.org)
[![macOS 14+](https://img.shields.io/badge/macOS-14%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Why

Most Swift LLM packages embed llama.cpp statically. If your app needs to
swap between an old and a new llama.cpp build to A/B test a regression,
or hosts another package that already bundles llama.cpp, you get symbol
collisions and a linker that refuses to cooperate.

This package solves it with two tricks:

1. **Protocol-first design.** `LlamaBackend` is the only thing your app
   imports; the concrete backend is injected at runtime.
2. **dlopen-based loading.** `NativeLlamaBackend` uses `dlopen()` +
   `dlsym()` to bind to `libllama.dylib` lazily, scoped with
   `RTLD_LOCAL`, so nothing leaks into the global symbol table.

## Backends shipped

| Backend | Status | Notes |
|---|---|---|
| `MockLlamaBackend` | ✅ stable | In-memory canned responses; used in tests |
| `LlamaCppBackend` | ✅ stable | Drives `llama-cli` (`brew install llama.cpp`) as a subprocess. Real streaming via chunked stdout |
| `NativeLlamaBackend` | 🧪 experimental | Direct `dlopen()` of `libllama.dylib`. No symbol collisions with embedded llama.cpp builds |

## Install

```swift
.package(url: "https://github.com/ArtemKyslicyn/Agent07ExperimentalLLM.git",
         from: "0.1.0")
```

Then in your target:

```swift
.product(name: "Agent07ExperimentalLLM", package: "Agent07ExperimentalLLM")
```

## Quick start

```swift
import Agent07ExperimentalLLM

// Pick a backend. The mock one needs nothing installed.
let backend = MockLlamaBackend(response: "Hello!")

// Or point at a real llama.cpp install:
// let backend = LlamaCppBackend(version: "b5200")  // resolves binary from PATH

let model = try await backend.loadModel(
    path: "/path/to/Qwen3-0.6B-Q4_K_M.gguf",
    config: .full
)

let result = try await backend.generate(
    model: model,
    prompt: "Write a haiku about Swift:",
    config: .creative
)

print(result.text)
print("\(result.tokensPerSecond) tok/s")
```

### Streaming

```swift
_ = try await backend.generateStream(
    model: model, prompt: "Stream me", config: .default
) { token in
    print(token, terminator: "")
}
```

### Layer specialization

`ModelLoadConfig` supports loading a subset of transformer layers — for
example, only the first 20 + last 5 for a "coding-shaped" model:

```swift
let config = ModelLoadConfig.codingLayers(totalLayers: 32)
let model = try await backend.loadModel(path: gguf, config: config)
```

Backends declare what they actually support via `BackendCapabilities`,
so callers can downgrade gracefully when a feature isn't available.

## Switching backends at runtime

```swift
let manager = BackendManager()
await manager.register(MockLlamaBackend(id: "test"))
await manager.register(LlamaCppBackend(version: "b5200"))

try await manager.switchTo(backendId: "llama.cpp-b5200")
let active = try await manager.active()
```

## Public surface

```
LlamaBackend                       — protocol every backend conforms to
BackendCapabilities                — feature flags per backend
ModelHandle                        — opaque handle returned by loadModel
ModelLoadConfig                    — .full / .cpuMinimal / .codingLayers(_:) / .reasoningLayers(_:)
GenerateConfig                     — .default / .creative / .precise
GenerateResult                     — text + tokenCount + duration + tokensPerSecond
ModelInfo                          — name, layers, parameters, ramUsage, layerSavings

MockLlamaBackend                   — in-memory backend for tests
LlamaCppBackend                    — subprocess driver for llama-cli
NativeLlamaBackend                 — dlopen-based libllama driver
BackendManager                     — registry + switcher

LlamaError                         — modelNotFound / binaryNotFound / modelNotLoaded
                                     / generationFailed / backendNotRegistered
```

## Testing

```bash
swift test
```

The test suite covers all three backends, the manager, model load configs,
generate-config presets, and the dynamic-library probe. Real llama-cli /
libllama installs are NOT required — the tests stub or skip those paths.

## Origin

Extracted from [Agent07](https://github.com/ArtemKyslicyn/Agent07) in
2026-05. Designed to be drop-in for any Swift project that wants to host
local LLM inference without committing to a single llama.cpp build.

## License

MIT — see [LICENSE](LICENSE).
