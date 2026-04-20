// ========== DOSYA: sentinel-inference/build.rs ==========
fn main() -> std::io::Result<()> {
    // tonic_build, .proto dosyalarını gRPC client/server koduna çevirir
    tonic_build::configure()
        .build_server(false) // Bu repo sadece istemci (client) olacak
        .compile(
            &["proto/market_data.proto", "proto/execution.proto", "proto/intelligence.proto"],
            &["proto/"] // Include yolu
        )?;
    Ok(())
}