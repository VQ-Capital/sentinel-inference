// ========== DOSYA: sentinel-inference/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use prost::Message;
use qdrant_client::qdrant::{Condition, Filter, Range, SearchPointsBuilder};
use qdrant_client::Qdrant;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{sleep, Duration};
use tracing::info;

pub mod sentinel_protos {
    pub mod market {
        include!(concat!(env!("OUT_DIR"), "/sentinel.market.v1.rs"));
    }
    pub mod execution {
        include!(concat!(env!("OUT_DIR"), "/sentinel.execution.v1.rs"));
    }
    pub mod intelligence {
        include!(concat!(env!("OUT_DIR"), "/sentinel.intelligence.v1.rs"));
    }
}

use sentinel_protos::execution::{trade_signal::SignalType, TradeSignal};
use sentinel_protos::intelligence::SemanticVector;
use sentinel_protos::market::{AggTrade, MarketStateVector, OrderbookDepth};

// -----------------------------------------------------------------------------
// 1. QUANT MATHEMATICS: EMA-Based Z-Score Engine
// -----------------------------------------------------------------------------
struct EmaZScore {
    mean: f64,
    variance: f64,
    alpha: f64,
    initialized: bool,
}

impl EmaZScore {
    fn new(window_size: usize) -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            alpha: 2.0 / (window_size as f64 + 1.0),
            initialized: false,
        }
    }

    fn update_and_get_z(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.mean = value;
            self.variance = 1.0;
            self.initialized = true;
            return 0.0;
        }

        let diff = value - self.mean;
        self.mean += self.alpha * diff;
        self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff);

        let std_dev = self.variance.sqrt();
        if std_dev < 1e-8 {
            0.0
        } else {
            (value - self.mean) / std_dev
        }
    }
}

struct SymbolFeatureNormalizer {
    velocity_z: EmaZScore,
    imbalance_z: EmaZScore,
    sentiment_z: EmaZScore,
}

impl SymbolFeatureNormalizer {
    fn new(window_size: usize) -> Self {
        Self {
            velocity_z: EmaZScore::new(window_size),
            imbalance_z: EmaZScore::new(window_size),
            sentiment_z: EmaZScore::new(window_size),
        }
    }
}

struct WindowStats {
    first_price: f64,
    last_price: f64,
    buy_volume: f64,
    sell_volume: f64,
    trade_count: i64,
    window_start_sec: i64,
}

// -----------------------------------------------------------------------------
// 3. MAIN RUNTIME
// -----------------------------------------------------------------------------
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());

    let window_size_sec: i64 = std::env::var("WINDOW_SIZE_SEC")
        .unwrap_or_else(|_| "30".to_string())
        .parse()
        .unwrap_or(30);
    let min_ticks: i64 = std::env::var("MIN_TICKS")
        .unwrap_or_else(|_| "25".to_string())
        .parse()
        .unwrap_or(25);
    let warmup_required: i32 = std::env::var("WARMUP_VECTORS")
        .unwrap_or_else(|_| "500".to_string())
        .parse()
        .unwrap_or(500);
    let similarity_threshold: f32 = std::env::var("SIMILARITY_THRESHOLD")
        .unwrap_or_else(|_| "0.985".to_string())
        .parse()
        .unwrap_or(0.985);
    let blindspot_sec: i64 = std::env::var("BLINDSPOT_SEC")
        .unwrap_or_else(|_| "900".to_string())
        .parse()
        .unwrap_or(900);

    let nats_client = async_nats::connect(&nats_url)
        .await
        .context("CRITICAL: NATS Bağlantı Hatası")?;

    let mut qdrant_client_opt = None;
    for _ in 1..=10 {
        if let Ok(client) = Qdrant::from_url(&qdrant_url).build() {
            if client.health_check().await.is_ok() {
                qdrant_client_opt = Some(client);
                info!("✅ Qdrant Vektör Veritabanı bağlandı.");
                break;
            }
        }
        sleep(Duration::from_secs(2)).await;
    }
    let qdrant_client = qdrant_client_opt.context("❌ Qdrant'a bağlanılamadı!")?;

    info!("🧠 Fusion Inference Motoru (ORDERBOOK + Z-SCORE) AKTİF!");

    // --- CACHE TANIMLAMALARI ---
    let sentiment_cache: Arc<RwLock<HashMap<String, (f64, i64)>>> =
        Arc::new(RwLock::new(HashMap::new()));
    let orderbook_cache: Arc<RwLock<HashMap<String, f64>>> = Arc::new(RwLock::new(HashMap::new()));

    // 1. NLP SENTIMENT DINLEYICISI
    let nats_intel_clone = nats_client.clone();
    let sentiment_cache_clone = sentiment_cache.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_intel_clone.subscribe("intelligence.news.vector").await {
            while let Some(msg) = sub.next().await {
                if let Ok(vector) = SemanticVector::decode(msg.payload) {
                    sentiment_cache_clone
                        .write()
                        .await
                        .insert(vector.symbol, (vector.sentiment_score, vector.timestamp));
                }
            }
        }
    });

    // 2. YENİ: L2 ORDERBOOK DINLEYICISI (Tahta Dengesizliği)
    let nats_ob_clone = nats_client.clone();
    let orderbook_cache_clone = orderbook_cache.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_ob_clone.subscribe("market.orderbook.>").await {
            info!("👀 Orderbook Depth (L2) dinleyicisi başlatıldı. Tahta analiz ediliyor.");
            while let Some(msg) = sub.next().await {
                if let Ok(depth) = OrderbookDepth::decode(msg.payload) {
                    let bids_vol: f64 = depth.bids.iter().map(|b| b.price * b.quantity).sum();
                    let asks_vol: f64 = depth.asks.iter().map(|a| a.price * a.quantity).sum();

                    let total_vol = bids_vol + asks_vol;
                    let ob_imbalance = if total_vol > 0.0 {
                        (bids_vol - asks_vol) / total_vol
                    } else {
                        0.0
                    };

                    orderbook_cache_clone
                        .write()
                        .await
                        .insert(depth.symbol, ob_imbalance);
                }
            }
        }
    });

    // 3. TRADE DINLEYICISI VE ANA FUSION LOOP
    let mut subscriber = nats_client.subscribe("market.trade.>").await?;
    let mut windows: HashMap<String, WindowStats> = HashMap::new();
    let mut normalizers: HashMap<String, SymbolFeatureNormalizer> = HashMap::new();
    let mut generated_vectors_count: HashMap<String, i32> = HashMap::new();

    while let Some(message) = subscriber.next().await {
        let trade = match AggTrade::decode(message.payload) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let trade_sec = trade.timestamp / 1000;
        let stats = windows.entry(trade.symbol.clone()).or_insert(WindowStats {
            first_price: trade.price,
            last_price: trade.price,
            buy_volume: 0.0,
            sell_volume: 0.0,
            trade_count: 0,
            window_start_sec: trade_sec,
        });

        if trade_sec >= stats.window_start_sec + window_size_sec {
            if stats.trade_count >= min_ticks {
                let price_velocity = (stats.last_price - stats.first_price) / stats.first_price;

                // TRADE IMBALANCE (Geçmiş işlemler)
                let total_trade_vol = stats.buy_volume + stats.sell_volume;
                let trade_imbalance = if total_trade_vol > 0.0 {
                    (stats.buy_volume - stats.sell_volume) / total_trade_vol
                } else {
                    0.0
                };

                // ORDERBOOK IMBALANCE (Gelecek Beklentisi)
                let ob_imbalance = *orderbook_cache
                    .read()
                    .await
                    .get(&trade.symbol)
                    .unwrap_or(&0.0);

                // 🔥 FUSION: Tahta derinliği %60, Gerçekleşen işlemler %40 ağırlıklandırıldı.
                let combined_imbalance = (trade_imbalance * 0.4) + (ob_imbalance * 0.6);

                let v_count = generated_vectors_count
                    .entry(trade.symbol.clone())
                    .or_insert(0);
                *v_count += 1;

                // NLP SENTIMENT
                let active_sentiment = {
                    let cache = sentiment_cache.read().await;
                    if let Some(&(score, ts)) = cache.get(&trade.symbol) {
                        if trade.timestamp - ts < 3600000 {
                            score
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                };

                // Z-SCORE NORMALIZATION
                let norm = normalizers
                    .entry(trade.symbol.clone())
                    .or_insert_with(|| SymbolFeatureNormalizer::new(1000));
                let z_velocity = norm.velocity_z.update_and_get_z(price_velocity);
                let z_imbalance = norm.imbalance_z.update_and_get_z(combined_imbalance);
                let z_sentiment = norm.sentiment_z.update_and_get_z(active_sentiment);

                // 3D FUSION VEKTÖRÜ
                let normalized_vector =
                    vec![z_velocity as f32, z_imbalance as f32, z_sentiment as f32];

                let max_allowed_timestamp = trade.timestamp - (blindspot_sec * 1000);
                let time_range = Range {
                    lt: Some(max_allowed_timestamp as f64),
                    ..Default::default()
                };
                let symbol_filter = Filter::all(vec![
                    Condition::matches("symbol", trade.symbol.clone()),
                    Condition::range("timestamp", time_range),
                ]);

                if *v_count < warmup_required {
                    if *v_count % 50 == 0 {
                        info!(
                            "🔥 WARM-UP [{}]: Vektörler Toplanıyor ({}/{})",
                            trade.symbol, v_count, warmup_required
                        );
                    }
                } else {
                    if *v_count == warmup_required {
                        info!(
                            "🚀 [WAKE UP] {} Keskin Nişancı Modu Devrede! Orderbook Motoru Aktif.",
                            trade.symbol
                        );
                    }

                    let mut final_signal = SignalType::Hold;
                    let mut confidence = 0.0;
                    let mut reason = "No match.".to_string();

                    if let Ok(response) = qdrant_client
                        .search_points(
                            SearchPointsBuilder::new("market_states", normalized_vector.clone(), 5)
                                .filter(symbol_filter.clone())
                                .with_payload(true),
                        )
                        .await
                    {
                        if let Some(best_match) = response
                            .result
                            .iter()
                            .find(|m| m.score >= similarity_threshold && m.score <= 0.999)
                        {
                            confidence = best_match.score as f64;

                            if z_velocity > 0.5 {
                                final_signal = SignalType::Buy;
                                reason = format!("Orderbook + Z-Score Match (Güven: {:.3}, İvme: {:.2}σ). Yön: YUKARI.", confidence, z_velocity);
                            } else if z_velocity < -0.5 {
                                final_signal = SignalType::Sell;
                                reason = format!("Orderbook + Z-Score Match (Güven: {:.3}, İvme: {:.2}σ). Yön: AŞAĞI.", confidence, z_velocity);
                            }
                        }
                    }

                    if final_signal != SignalType::Hold {
                        let trade_signal = TradeSignal {
                            symbol: trade.symbol.clone(),
                            r#type: final_signal.into(),
                            confidence_score: confidence,
                            recommended_leverage: 1.0,
                            timestamp: trade.timestamp,
                            reason,
                        };
                        let mut sig_buf = Vec::new();
                        if trade_signal.encode(&mut sig_buf).is_ok() {
                            let _ = nats_client
                                .publish(format!("signal.trade.{}", trade.symbol), sig_buf.into())
                                .await;
                        }
                    }
                }

                let current_state = MarketStateVector {
                    symbol: trade.symbol.clone(),
                    window_start_time: stats.window_start_sec * 1000,
                    window_end_time: trade_sec * 1000,
                    price_velocity,
                    volume_imbalance: combined_imbalance, // Artık L2 Derinliği içeriyor
                    sentiment_score: active_sentiment,
                    embeddings: vec![z_velocity, z_imbalance, z_sentiment],
                };
                let mut state_buf = Vec::new();
                if current_state.encode(&mut state_buf).is_ok() {
                    let _ = nats_client
                        .publish(format!("state.vector.{}", trade.symbol), state_buf.into())
                        .await;
                }
            }

            stats.trade_count = 0;
            stats.first_price = trade.price;
            stats.buy_volume = 0.0;
            stats.sell_volume = 0.0;
            stats.window_start_sec = trade_sec;
        }

        stats.last_price = trade.price;
        stats.trade_count += 1;
        if trade.is_buyer_maker {
            stats.sell_volume += trade.quantity;
        } else {
            stats.buy_volume += trade.quantity;
        }
    }

    Ok(())
}
