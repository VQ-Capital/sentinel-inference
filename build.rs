fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().build_server(false).compile(
        &[
            "sentinel-spec/proto/sentinel/market/v1/market_data.proto",
            "sentinel-spec/proto/sentinel/execution/v1/execution.proto",
            "sentinel-spec/proto/sentinel/intelligence/v1/intelligence.proto",
        ],
        &["sentinel-spec/proto/"],
    )?;
    Ok(())
}
