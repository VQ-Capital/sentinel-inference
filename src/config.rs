// ========== DOSYA: sentinel-inference/src/config.rs ==========
#[derive(Clone, Debug)]
pub struct AppConfig {
    pub nats_url: String,
    pub qdrant_url: String,
    pub qdrant_collection: String,
    pub min_ticks: i64,
    pub warmup_vectors: i32,
    pub z_scale: f64,
    pub ai_timeout_ms: u64,
    pub min_confidence_score: f64,
    pub min_signal_interval_ms: i64,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            nats_url: std::env::var("NATS_URL")
                .unwrap_or_else(|_| "nats://localhost:4222".to_string()),
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),
            qdrant_collection: std::env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "market_states_12d".to_string()),
            min_ticks: std::env::var("MIN_TICKS")
                .unwrap_or_else(|_| "25".to_string())
                .parse()
                .expect("ENV ERROR: MIN_TICKS"),
            warmup_vectors: std::env::var("WARMUP_VECTORS")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .expect("ENV ERROR: WARMUP_VECTORS"),
            z_scale: std::env::var("Z_SCALE")
                .unwrap_or_else(|_| "10000.0".to_string())
                .parse()
                .expect("ENV ERROR: Z_SCALE"),
            ai_timeout_ms: std::env::var("AI_TIMEOUT_MS")
                .unwrap_or_else(|_| "25".to_string())
                .parse()
                .expect("ENV ERROR: AI_TIMEOUT_MS"),
            min_confidence_score: std::env::var("MIN_CONFIDENCE_SCORE")
                .unwrap_or_else(|_| "0.65".to_string())
                .parse()
                .expect("ENV ERROR: MIN_CONFIDENCE_SCORE"),
            min_signal_interval_ms: std::env::var("MIN_SIGNAL_INTERVAL_MS")
                .unwrap_or_else(|_| "15000".to_string())
                .parse()
                .expect("ENV ERROR: MIN_SIGNAL_INTERVAL_MS"),
        }
    }
}
