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
// YENİ İSİMLER BURADA
use sentinel_protos::intelligence::sentiment_analyzer_service_client::SentimentAnalyzerServiceClient;
use sentinel_protos::intelligence::AnalyzeTextRequest;
use sentinel_protos::market::{AggTrade, MarketStateVector};

struct WindowStats {
    first_price: f64,
    last_price: f64,
    buy_volume: f64,
    sell_volume: f64,
    trade_count: i64,
    current_window_sec: i64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    let intel_url =
        std::env::var("INTELLIGENCE_URL").unwrap_or_else(|_| "http://localhost:50051".to_string());

    let nats_client = async_nats::connect(&nats_url)
        .await
        .context("NATS Bağlantı Hatası")?;

    let mut qdrant_client = None;
    for i in 1..=10 {
        if let Ok(client) = Qdrant::from_url(&qdrant_url).build() {
            if client.health_check().await.is_ok() {
                qdrant_client = Some(client);
                info!("✅ Qdrant bağlandı.");
                break;
            }
        }
        warn!("⚠️ Qdrant aranıyor... ({}/10)", i);
        sleep(Duration::from_secs(2)).await;
    }
    let qdrant_client = qdrant_client.context("❌ Qdrant'a bağlanılamadı!")?;

    // gRPC Intelligence Client (YENİ İSİM)
    let mut intel_client = match SentimentAnalyzerServiceClient::connect(intel_url.clone()).await {
        Ok(c) => Some(c),
        Err(e) => {
            warn!(
                "⚠️ Intelligence servisine bağlanılamadı (NLP Devre Dışı): {}",
                e
            );
            None
        }
    };

    let mut subscriber = nats_client
        .subscribe("market.trade.>")
        .await
        .context("NATS Kanal Hatası")?;
    let mut windows: HashMap<String, WindowStats> = HashMap::new();

    info!("🧠 Inference Motoru AKTİF: 3D Vektör Arama Modu Çalışıyor.");

    while let Some(message) = subscriber.next().await {
        if let Ok(trade) = AggTrade::decode(message.payload) {
            let trade_sec = trade.timestamp / 1000;

            let stats = windows.entry(trade.symbol.clone()).or_insert(WindowStats {
                first_price: trade.price,
                last_price: trade.price,
                buy_volume: 0.0,
                sell_volume: 0.0,
                trade_count: 0,
                current_window_sec: trade_sec,
            });

            if trade_sec > stats.current_window_sec {
                if stats.trade_count > 0 {
                    let price_velocity = (stats.last_price - stats.first_price) / stats.first_price;
                    let total_vol = stats.buy_volume + stats.sell_volume;
                    let volume_imbalance = if total_vol > 0.0 {
                        (stats.buy_volume - stats.sell_volume) / total_vol
                    } else {
                        0.0
                    };

                    let simulated_news = if price_velocity > 0.0001 {
                        "Massive breakout seen, market looks highly bullish and ready to moon!"
                    } else if price_velocity < -0.0001 {
                        "Market faces heavy selloff, bearish trend indicates a crash and dump."
                    } else {
                        "Market is ranging near support, accumulation ongoing."
                    };

                    let mut sentiment_score = 0.0;
                    if let Some(client) = &mut intel_client {
                        // YENİ REQUEST İSMİ BURADA
                        let req = tonic::Request::new(AnalyzeTextRequest {
                            text: simulated_news.into(),
                        });
                        if let Ok(resp) = client.analyze_text(req).await {
                            sentiment_score = resp.into_inner().score;
                        }
                    }

                    let current_vector = vec![
                        price_velocity as f32,
                        volume_imbalance as f32,
                        sentiment_score as f32,
                    ];

                    let mut final_signal = SignalType::Hold;
                    let mut confidence = 0.5;
                    let mut reason = "No historical match. Holding.".to_string();

                    let search_result = qdrant_client
                        .search_points(
                            SearchPointsBuilder::new("market_states", current_vector.clone(), 1)
                                .with_payload(true),
                        )
                        .await;

                    match search_result {
                        Ok(response) => {
                            if let Some(best_match) = response.result.first() {
                                if best_match.score > 0.95 {
                                    let past_velocity = best_match
                                        .payload
                                        .get("velocity")
                                        .and_then(|v| v.as_double())
                                        .unwrap_or(0.0);

                                    confidence = best_match.score as f64;

                                    if past_velocity > 0.0 {
                                        final_signal = SignalType::Buy;
                                        reason = format!(
                                            "Vektör Eşleşmesi (Skor: {:.2}). Geçmiş trend: Yukarı.",
                                            confidence
                                        );
                                    } else {
                                        final_signal = SignalType::Sell;
                                        reason = format!(
                                            "Vektör Eşleşmesi (Skor: {:.2}). Geçmiş trend: Aşağı.",
                                            confidence
                                        );
                                    }
                                    info!(
                                        "🚀 [AI BEYNİ] {} -> Sinyal: {:?} | Neden: {}",
                                        trade.symbol, final_signal, reason
                                    );
                                }
                            }
                        }
                        Err(e) => warn!("Qdrant Arama Hatası: {}", e),
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

                    let current_state = MarketStateVector {
                        symbol: trade.symbol.clone(),
                        window_start_time: 0,
                        window_end_time: 0,
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
                stats.current_window_sec = trade_sec;
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
