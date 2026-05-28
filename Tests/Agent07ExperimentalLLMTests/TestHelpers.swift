import Foundation

/// Lock-backed array for tests that need to capture values from synchronous
/// `@Sendable` closures. Swift 6 disallows mutation of captured vars; this
/// wrapper provides the same ergonomics safely.
final class LockedArray<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [T] = []
    func append(_ item: T) { lock.lock(); defer { lock.unlock() }; storage.append(item) }
    var snapshot: [T] { lock.lock(); defer { lock.unlock() }; return storage }
}
