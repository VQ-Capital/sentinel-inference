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
use sentinel_protos::market::{AggTrade, MarketStateVector};

struct WindowStats {
    first_price: f64,
    last_price: f64,
    buy_volume: f64,
    sell_volume: f64,
    trade_count: i64,
    window_start_sec: i64,
}

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

    let mut qdrant_client = None;
    for _i in 1..=10 {
        if let Ok(client) = Qdrant::from_url(&qdrant_url).build() {
            if client.health_check().await.is_ok() {
                qdrant_client = Some(client);
                info!("✅ Qdrant Vektör Veritabanı bağlandı.");
                break;
            }
        }
        sleep(Duration::from_secs(2)).await;
    }
    let qdrant_client = qdrant_client.context("❌ Qdrant'a bağlanılamadı!")?;

    info!("🧠 Fusion Inference Motoru (QUANT V3) AKTİF | Sözel + Sayısal Veri Birleştiriliyor...");

    // YENİ: SENTIMENT CACHE (Haber Duygularını Hafızada Tut)
    // Değer: (Sentiment Skoru, Haber Zamanı)
    let sentiment_cache: Arc<RwLock<HashMap<String, (f64, i64)>>> =
        Arc::new(RwLock::new(HashMap::new()));

    let nats_intel_clone = nats_client.clone();
    let sentiment_cache_clone = sentiment_cache.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_intel_clone.subscribe("intelligence.news.vector").await {
            info!("👂 Fusion Node: Semantic (Haber) vektörlerini dinlemeye başladı.");
            while let Some(msg) = sub.next().await {
                if let Ok(vector) = SemanticVector::decode(msg.payload) {
                    let mut cache = sentiment_cache_clone.write().await;
                    cache.insert(vector.symbol, (vector.sentiment_score, vector.timestamp));
                }
            }
        }
    });

    let mut subscriber = nats_client.subscribe("market.trade.>").await?;
    let mut windows: HashMap<String, WindowStats> = HashMap::new();
    let mut generated_vectors_count: HashMap<String, i32> = HashMap::new();

    while let Some(message) = subscriber.next().await {
        if let Ok(trade) = AggTrade::decode(message.payload) {
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
                    let total_vol = stats.buy_volume + stats.sell_volume;
                    let volume_imbalance = if total_vol > 0.0 {
                        (stats.buy_volume - stats.sell_volume) / total_vol
                    } else {
                        0.0
                    };

                    let v_count = generated_vectors_count
                        .entry(trade.symbol.clone())
                        .or_insert(0);
                    *v_count += 1;

                    // YENİ: TIME-DECAY NLP FUSION
                    let active_sentiment = {
                        let cache = sentiment_cache.read().await;
                        if let Some(&(score, ts)) = cache.get(&trade.symbol) {
                            // Haber 1 saatten (3600000 ms) eskiyse etkisini sıfırla
                            if trade.timestamp - ts < 3600000 {
                                score
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        }
                    };

                    // MATEMATİKSEL DÜZELTME: Fiyat ivmesi Qdrant'ta Cosine etkisini kaybetmesin diye x10.000 ölçeklendi
                    let scaled_velocity = (price_velocity * 10000.0) as f32;

                    // 3D FUSION VEKTÖRÜ: [Fiyat İvmesi, Hacim Dengesizliği, Duygu Skoru]
                    let current_vector = vec![
                        scaled_velocity,
                        volume_imbalance as f32,
                        active_sentiment as f32,
                    ];

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
                                "🔥 WARM-UP [{}]: Yeni Vektörler Toplanıyor ({}/{})",
                                trade.symbol, v_count, warmup_required
                            );
                        }
                    } else {
                        if *v_count == warmup_required {
                            info!(
                                "🚀 [WAKE UP] {} Keskin Nişancı Modu Devrede! Av Bekleniyor...",
                                trade.symbol
                            );
                        }

                        let mut final_signal = SignalType::Hold;
                        let mut confidence = 0.0;
                        let mut reason = "No match.".to_string();

                        if let Ok(response) = qdrant_client
                            .search_points(
                                SearchPointsBuilder::new(
                                    "market_states",
                                    current_vector.clone(),
                                    5,
                                )
                                .filter(symbol_filter.clone())
                                .with_payload(true),
                            )
                            .await
                        {
                            let mut match_found = false;

                            if let Some(best_match) = response
                                .result
                                .iter()
                                .find(|m| m.score >= similarity_threshold && m.score <= 0.999)
                            {
                                match_found = true;
                                confidence = best_match.score as f64;

                                // Qdrant'taki ölçeklenmiş velocity'i normale çevirerek hesapla
                                let past_scaled_velocity = best_match
                                    .payload
                                    .get("velocity")
                                    .and_then(|v| v.as_double())
                                    .unwrap_or(0.0);
                                let past_velocity = past_scaled_velocity / 10000.0;
                                let past_timestamp = best_match
                                    .payload
                                    .get("timestamp")
                                    .and_then(|v| v.as_integer())
                                    .unwrap_or(0);
                                let minutes_ago = (trade.timestamp - past_timestamp) / 60000;

                                // Eğer hem geçmiş fiyat ivmesi hem de güncel fiyat uyumluysa vur!
                                if past_velocity > 0.0002 && price_velocity >= -0.0001 {
                                    final_signal = SignalType::Buy;
                                    reason = format!("Fusion Match ({} dk önce, Güven: {:.3}, Duygu: {:.1}). Yön: YUKARI.", minutes_ago, confidence, active_sentiment);
                                } else if past_velocity < -0.0002 && price_velocity <= 0.0001 {
                                    final_signal = SignalType::Sell;
                                    reason = format!("Fusion Match ({} dk önce, Güven: {:.3}, Duygu: {:.1}). Yön: AŞAĞI.", minutes_ago, confidence, active_sentiment);
                                }
                            }

                            if !match_found && *v_count % 20 == 0 {
                                if let Some(top_match) = response.result.first() {
                                    let mins_ago = (trade.timestamp
                                        - top_match
                                            .payload
                                            .get("timestamp")
                                            .and_then(|v| v.as_integer())
                                            .unwrap_or(0))
                                        / 60000;
                                    info!("🔭 [RADAR] {} (Duygu: {:.2}) Taranıyor. En iyi GEÇMİŞ: {:.3} ({} dk önce)",
                                        trade.symbol, active_sentiment, top_match.score, mins_ago);
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
                                    .publish(
                                        format!("signal.trade.{}", trade.symbol),
                                        sig_buf.into(),
                                    )
                                    .await;
                            }
                        }
                    }

                    // Vektörü NATS'a bas (Storage bunu alıp Qdrant'a yazacak)
                    // DİKKAT: price_velocity'i skalanmış haliyle gönderiyoruz ki Storage direkt yazsın ve arama eşleşsin.
                    let current_state = MarketStateVector {
                        symbol: trade.symbol.clone(),
                        window_start_time: stats.window_start_sec * 1000,
                        window_end_time: trade_sec * 1000,
                        price_velocity: scaled_velocity as f64,
                        volume_imbalance,
                        sentiment_score: active_sentiment,
                        embeddings: vec![],
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
    }
    Ok(())
}
