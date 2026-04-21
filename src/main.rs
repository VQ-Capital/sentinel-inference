// ========== DOSYA: sentinel-inference/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use prost::Message;
use qdrant_client::qdrant::SearchPointsBuilder;
use qdrant_client::Qdrant;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
use tracing::{info, warn};

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
use sentinel_protos::intelligence::sentiment_analyzer_service_client::SentimentAnalyzerServiceClient;
use sentinel_protos::intelligence::AnalyzeTextRequest;
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

    // Parametrik ENV Değişkenleri
    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    let intel_url =
        std::env::var("INTELLIGENCE_URL").unwrap_or_else(|_| "http://localhost:50051".to_string());

    let window_size_sec: i64 = std::env::var("WINDOW_SIZE_SEC")
        .unwrap_or_else(|_| "30".to_string())
        .parse()
        .unwrap_or(30);
    let min_ticks: i64 = std::env::var("MIN_TICKS")
        .unwrap_or_else(|_| "25".to_string())
        .parse()
        .unwrap_or(25);
    let warmup_required: i32 = std::env::var("WARMUP_VECTORS")
        .unwrap_or_else(|_| "50".to_string())
        .parse()
        .unwrap_or(50);
    let similarity_threshold: f32 = std::env::var("SIMILARITY_THRESHOLD")
        .unwrap_or_else(|_| "0.97".to_string())
        .parse()
        .unwrap_or(0.97);

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

    let mut intel_client = match SentimentAnalyzerServiceClient::connect(intel_url.clone()).await {
        Ok(c) => Some(c),
        Err(e) => {
            warn!("⚠️ NLP Devre Dışı: {}", e);
            None
        }
    };

    let mut subscriber = nats_client.subscribe("market.trade.>").await?;
    let mut windows: HashMap<String, WindowStats> = HashMap::new();

    let mut generated_vectors_count = 0;

    info!(
        "🧠 Inference Motoru AKTİF | Window: {}s | Warmup: {} vektör | Threshold: {}",
        window_size_sec, warmup_required, similarity_threshold
    );

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

                    let mut sentiment_score = 0.0;
                    if let Some(client) = &mut intel_client {
                        let simulated_news = if price_velocity > 0.0005 {
                            "Massive breakout seen, highly bullish!"
                        } else if price_velocity < -0.0005 {
                            "Market faces heavy selloff, dump."
                        } else {
                            "Market is ranging near support."
                        };
                        let req = tonic::Request::new(AnalyzeTextRequest {
                            text: simulated_news.into(),
                        });
                        if let Ok(resp) = client.analyze_text(req).await {
                            sentiment_score = resp.into_inner().score;
                        }
                    }

                    generated_vectors_count += 1;
                    let current_vector = vec![
                        price_velocity as f32,
                        volume_imbalance as f32,
                        sentiment_score as f32,
                    ];

                    // COLD START (ISINMA) KONTROLÜ
                    if generated_vectors_count < warmup_required {
                        if generated_vectors_count % 10 == 0 {
                            info!("🔥 WARM-UP SÜRECİ: Vektörler toplanıyor ({}/{}) İşlem yapılmayacak.", generated_vectors_count, warmup_required);
                        }
                    } else {
                        // Isınma Bitti, İşlem Arayabiliriz
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
                                .with_payload(true),
                            )
                            .await
                        {
                            // DİNAMİK THRESHOLD KULLANILDI (Örn: > 0.97)
                            if let Some(best_match) = response
                                .result
                                .into_iter()
                                .find(|m| m.score >= similarity_threshold && m.score <= 0.995)
                            {
                                let past_velocity = best_match
                                    .payload
                                    .get("velocity")
                                    .and_then(|v| v.as_double())
                                    .unwrap_or(0.0);
                                confidence = best_match.score as f64;

                                if past_velocity > 0.0002 {
                                    final_signal = SignalType::Buy;
                                    reason = format!(
                                        "Vektör Eşleşmesi (Skor: {:.3}). Geçmiş trend: Yukarı.",
                                        confidence
                                    );
                                } else if past_velocity < -0.0002 {
                                    final_signal = SignalType::Sell;
                                    reason = format!(
                                        "Vektör Eşleşmesi (Skor: {:.3}). Geçmiş trend: Aşağı.",
                                        confidence
                                    );
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

                    // Vektörü her halükarda kaydet (Storage için)
                    let current_state = MarketStateVector {
                        symbol: trade.symbol.clone(),
                        window_start_time: stats.window_start_sec * 1000,
                        window_end_time: trade_sec * 1000,
                        price_velocity,
                        volume_imbalance,
                        sentiment_score,
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
