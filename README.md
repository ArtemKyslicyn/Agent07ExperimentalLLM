# Agent07ExperimentalLLM

A pluggable, runtime-loadable LLM backend layer for Swift. Switch between
llama.cpp builds, a subprocess driver, an in-memory mock, and (foundation
laid for) direct `dlopen` of `libllama` — without recompiling, and without
symbol collisions when multiple llama.cpp variants live in the same process.

[![CI](https://github.com/ArtemKyslicyn/Agent07ExperimentalLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/ArtemKyslicyn/Agent07ExperimentalLLM/actions/workflows/ci.yml)
[![Swift 6](https://img.shields.io/badge/Swift-6.0-orange.svg)](https://swift.org)
[![macOS 14+](https://img.shields.io/badge/macOS-14%2B-blue.svg)](#)
[![Tests](https://img.shields.io/badge/Tests-47%20passing-success.svg)](#testing)
[![Coverage](https://img.shields.io/badge/Coverage-44%25-orange.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## About

`Agent07ExperimentalLLM` is the **experimental backend bench** of the
[Agent07](https://github.com/ArtemKyslicyn/Agent07) ecosystem — an
open-core macOS app for building local-LLM agent pipelines. It is a small,
self-contained Swift 6 package that defines a single inference protocol,
`LlamaBackend`, and ships several interchangeable implementations behind it.

It is deliberately kept **off the production inference path**. Agent07's
shipping inference runs through `Agent07LLMServices` (a wrapper around
LLM.swift / llama.cpp embedded statically). This package exists so that
alternative or in-flight backends — a `llama-cli` subprocess driver, a
`dlopen`-based native loader, a deterministic mock — can be developed,
A/B-tested, and benchmarked **without touching, relinking, or risking the
production engine**. That separation is exactly why it is labelled
"experimental": the backends here are pluggable and decoupled, free to
target newer llama.cpp builds or unproven loading strategies that you would
not want statically linked into the main app.

Because it has **zero dependencies** (Foundation + `os.log` only) and no
system-library (`CLlama`) target, it is also a clean drop-in for any Swift
project that wants to host local inference behind a protocol without
committing to one llama.cpp build.

### Why pluggable backends?

Most Swift LLM packages embed llama.cpp statically. If your app needs to
A/B an old vs. a new llama.cpp build to chase a regression, or it already
hosts another package that bundles llama.cpp, you get duplicate symbols and
a linker that refuses to cooperate. This package sidesteps that two ways:

1. **Protocol-first design.** `LlamaBackend` is the only type your app
   imports; the concrete backend is injected and swapped at runtime through
   `BackendManager`.
2. **`dlopen`-based loading.** `NativeLlamaBackend` binds to
   `libllama.dylib` lazily via `dlopen()` + `dlsym()`, scoped with
   `RTLD_LOCAL`, so nothing leaks into the global symbol table and nothing
   collides with an already-embedded llama.cpp.

## Requirements

| | |
|---|---|
| Platform | macOS 14+ |
| Toolchain | Swift 6.0 (language mode v6) |
| Dependencies | none (Foundation, `os.log`) |
| Optional runtime | `brew install llama.cpp` — only for the two real backends |

## Install

```swift
.package(url: "https://github.com/ArtemKyslicyn/Agent07ExperimentalLLM.git",
         from: "0.1.0")
```

Then add the product to your target:

```swift
.product(name: "Agent07ExperimentalLLM", package: "Agent07ExperimentalLLM")
```

## Backends shipped

Every backend conforms to `LlamaBackend` and is therefore interchangeable.

| Backend | Status | Mechanism |
|---|---|---|
| `MockLlamaBackend` | ✅ stable | In-memory canned response. No I/O, no install. Used in tests and previews. |
| `LlamaCppBackend` | ✅ stable | Spawns the `llama-cli` / `llama-completion` binary as a subprocess and reads stdout. |
| `NativeLlamaBackend` | 🧪 experimental | `dlopen()`s `libllama.dylib`, resolves symbols for model/context queries; **generation currently delegates to `LlamaCppBackend`**. |

### `MockLlamaBackend`

A deterministic, dependency-free backend. `loadModel` returns a handle
without reading anything; `generate` / `generateStream` return a fixed
string supplied at init; `modelInfo` reports stub metadata (1M params,
4 layers). Capabilities are configurable, so it can stand in for any real
backend's feature flags in tests.

```swift
let backend = MockLlamaBackend(id: "test", response: "Hello!",
                               capabilities: .init(realStreaming: true))
```

### `LlamaCppBackend` (subprocess driver)

Drives a llama.cpp command-line binary out of process. The binary is
auto-discovered (preferring the non-interactive `llama-completion`, then
`llama-cli`) across Homebrew, `/usr/local`, and `~/.local` / `~/llama.cpp`
build locations — or you can pin an exact `binaryPath`. The `version` tag
is folded into `backendId` as `llama.cpp-<version>`, which is how you
register two different builds side by side.

`loadModel` validates that both the GGUF and the binary exist (throwing
`LlamaError.modelNotFound` / `.binaryNotFound`) and **remembers the
caller's `ModelLoadConfig` per handle**, so generation uses the real
context window you asked for rather than a hardcoded default.

Each `generate` call assembles `llama-cli` arguments from the configs:

- `-m`, `-p`, `-n <maxTokens>`, `--temp`, `--top-p`, `--top-k`,
  `--repeat-penalty`, `-t <threads>`
- `--no-display-prompt` and `-no-cnv` (Homebrew b8680+ defaults to
  conversation mode, which yields no usable stdout for a one-shot `-p`)
- `-ngl` from `gpuLayers` (`-1` → offload all, mapped to `99`),
  `-c <contextSize>`, and an approximate layer count from `layerMask`
- **Adaptive timeout** scaled to the token budget
  (`min(1800, 120 + maxTokens/8)` seconds) so large/reasoning models are
  not killed mid-turn; partial output is still returned if the process is
  terminated.
- **Post-hoc stop sequences:** `--stop` is rejected by newer Homebrew
  builds, so `stopSequences` are applied by truncating stdout at the
  earliest marker.

> **Streaming note:** `capabilities.realStreaming` is `true`, but because a
> live `readabilityHandler` proved unreliable inside GUI apps,
> `generateStream` runs the same blocking subprocess and then replays the
> output in ~24 chunks through `onToken`. Treat it as chunked, not
> token-exact.

You can also inject `environmentExtras`, merged into the subprocess
environment (e.g. Agent07 model-directory variables).

### `NativeLlamaBackend` (`dlopen` foundation)

Loads `libllama.dylib` lazily on first `loadModel`, with `RTLD_NOW |
RTLD_LOCAL` so its symbols never collide with a statically embedded
llama.cpp. The internal `LlamaDynLib` resolves a focused set of C entry
points via `dlsym` — `llama_backend_init`, `llama_model_load_from_file`,
`llama_model_free`, `llama_model_n_layer`, `llama_init_from_model`,
`llama_free`, `llama_n_ctx` — each bound as a typed `@convention(c)`
function pointer, and `dlclose`s on `deinit`. `LlamaDynLib.isAvailable`
(and `NativeLlamaBackend.isAvailable`) probe the filesystem without loading
anything.

This is the **foundation**: the library is opened and the model can be
queried, but full native generation would require resolving ~20 more
symbols (tokenize, decode, sampler/batch APIs). Until then,
`generate` / `generateStream` **delegate to a `LlamaCppBackend` subprocess**
so generation stays safe and collision-free.

## Quick start

```swift
import Agent07ExperimentalLLM

// Pick a backend. The mock one needs nothing installed.
let backend = MockLlamaBackend(response: "Hello!")

// Or drive a real llama.cpp install (binary auto-discovered from PATH):
// let backend = LlamaCppBackend(version: "b5200")

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

`ModelLoadConfig` can request a subset of transformer layers — e.g. only
the first 20 + last 5 for a "coding-shaped" model:

```swift
let config = ModelLoadConfig.codingLayers(totalLayers: 32)
let model = try await backend.loadModel(path: gguf, config: config)
```

`ModelInfo.layerSavings` reports the resulting RAM reduction (0.0–1.0).
Backends advertise what they actually support via `BackendCapabilities`, so
callers can downgrade gracefully when a feature isn't available — the
subprocess backend, for instance, can only *approximate* a layer mask.

## Switching backends at runtime

`BackendManager` is an `actor` registry. The first backend registered
becomes active automatically.

```swift
let manager = BackendManager()
await manager.register(MockLlamaBackend(id: "test"))
await manager.register(LlamaCppBackend(version: "b5200"))  // id: llama.cpp-b5200
// convenience: await manager.registerLlamaCpp(version: "b5200")

try await manager.switchTo(backendId: "llama.cpp-b5200")
let active = try await manager.active()

let ids = await manager.registeredBackends          // sorted ids
let matrix = await manager.capabilityMatrix         // [(id, BackendCapabilities)]
```

`switchTo` and `active()` throw `BackendManagerError`
(`.backendNotFound` / `.noActiveBackend`).

## Public surface

```
LlamaBackend            — protocol every backend conforms to (Sendable)
                          loadModel / generate / generateStream / unloadModel / modelInfo
BackendCapabilities     — realStreaming, layerSpecialization, speculativeDecoding,
                          kvCacheControl, metalSupport, maxContext
ModelHandle             — opaque, Identifiable handle returned by loadModel
ModelLoadConfig         — gpuLayers / contextSize / layerMask / quantization / threads / mmap
                          presets: .full / .cpuMinimal
                          factories: .codingLayers(totalLayers:) / .reasoningLayers(totalLayers:)
GenerateConfig          — maxTokens / temperature / topP / topK / repeatPenalty / stopSequences
                          presets: .default / .creative / .precise
GenerateResult          — text + tokenCount + duration + tokensPerSecond (computed)
ModelInfo               — name, parameterCount, layerCount, contextSize, quantization,
                          fileSizeBytes, loadedLayers, ramUsageBytes, layerSavings (computed)

MockLlamaBackend        — in-memory backend for tests/previews
LlamaCppBackend         — subprocess driver for llama-cli / llama-completion
NativeLlamaBackend      — dlopen-based libllama loader (generation via CLI fallback)
BackendManager          — actor registry + runtime switcher

LlamaError              — modelNotFound / binaryNotFound / modelNotLoaded
                          / processError(exitCode:stderr:) / layerMaskNotSupported
BackendManagerError     — backendNotFound / noActiveBackend
```

## Testing

```bash
swift test
```

47 tests (`swift-testing`) cover the value types (`BackendCapabilities`,
`ModelHandle`, `ModelInfo` savings math, the config presets and custom
inits, `GenerateResult` tok/s), the `MockLlamaBackend` contract, the
`BackendManager` actor (register / switch / active / capability matrix /
error variants), the `LlamaCppBackend` surface (capabilities, id, argument
guards, `modelNotFound` / `binaryNotFound` / `modelNotLoaded` paths,
environment extras), and the `LlamaDynLib` probe + wrong-path load.

Real `llama-cli` / `libllama` installs are **not required** — tests that
would touch them use fake paths, stub binaries (e.g. `/bin/echo`), or
filesystem probes that accept either outcome, so the suite runs fully
hermetically in CI. CI additionally runs `swift test --enable-code-coverage`
and auto-updates the Coverage badge above.

## Origin

Extracted from [Agent07](https://github.com/ArtemKyslicyn/Agent07) in
2026-05 as part of its open-core split. Designed to be drop-in for any
Swift project that wants to host local LLM inference behind a protocol
without committing to a single llama.cpp build.

## License

MIT — see [LICENSE](LICENSE).
</content>
</invoke>
