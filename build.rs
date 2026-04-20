fn main() -> std::io::Result<()> {
    prost_build::compile_protos(
        &["proto/market_data.proto", "proto/execution.proto"],
        &["proto/"]
    )?;
    Ok(())
}