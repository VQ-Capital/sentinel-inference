// ========== DOSYA: sentinel-inference/build.rs ==========
use std::io::Result;

fn main() -> Result<()> {
    // Cargo'ya, proto dosyası değişirse projeyi yeniden derlemesini söyleriz
    println!("cargo:rerun-if-changed=proto/market_data.proto");
    
    // market_data.proto dosyasını derler ve Rust kodu üretir
    prost_build::compile_protos(&["proto/market_data.proto"], &["proto/"])?;
    Ok(())
}