// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Momij",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MomijCore", targets: ["MomijCore"]),
        .executable(name: "momij", targets: ["momij"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.0.0"),
        .package(url: "https://github.com/mattt/swift-xgrammar.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "MomijCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "XGrammar", package: "swift-xgrammar"),
            ]
        ),
        .executableTarget(
            name: "momij",
            dependencies: [
                "MomijCore",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "XGrammar", package: "swift-xgrammar"),
            ]
        ),
        .testTarget(
            name: "MomijCoreTests",
            dependencies: [
                "MomijCore",
                .product(name: "XGrammar", package: "swift-xgrammar"),
            ]
        ),
    ]
)
