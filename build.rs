// ========== DOSYA: sentinel-inference/build.rs ==========
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        .compile(
            &["proto/market_data.proto", "proto/execution.proto", "proto/intelligence.proto"],
            &["proto/"]
        )?;
    Ok(())
}