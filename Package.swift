// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Agent07ExperimentalLLM",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "Agent07ExperimentalLLM", targets: ["Agent07ExperimentalLLM"])
    ],
    targets: [
        // No CLlama system library — uses dlopen() at runtime to load
        // libllama.dylib lazily and avoid symbol collision with other
        // embedded llama.cpp builds (e.g. LLM.swift).
        .target(name: "Agent07ExperimentalLLM"),
        .testTarget(
            name: "Agent07ExperimentalLLMTests",
            dependencies: ["Agent07ExperimentalLLM"]
        )
    ],
    swiftLanguageModes: [.v6]
)
