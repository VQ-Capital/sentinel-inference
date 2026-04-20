// ========== DOSYA: sentinel-inference/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use prost::Message;
use std::collections::HashMap;
use tokio::time::{self, Duration};
use tracing::{info, error};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::SearchPointsBuilder;
use chrono;

// Protobuf modül tanımları
pub mod sentinel_protos {
    pub mod market {
        include!(concat!(env!("OUT_DIR"), "/sentinel.market.rs"));
    }
    pub mod execution {
        include!(concat!(env!("OUT_DIR"), "/sentinel.execution.rs"));
    }
    pub mod intelligence {
        include!(concat!(env!("OUT_DIR"), "/sentinel.intelligence.rs"));
    }
}

use sentinel_protos::market::{AggTrade, MarketStateVector};
use sentinel_protos::execution::{TradeSignal, trade_signal::SignalType};
use sentinel_protos::intelligence::sentiment_analyzer_client::SentimentAnalyzerClient;
use sentinel_protos::intelligence::SentimentRequest;

struct WindowStats {
    first_price: f64,
    last_price: f64,
    buy_volume: f64,
    sell_volume: f64,
    trade_count: i64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("🧠 Sentinel-Inference (Zeka Motoru) başlatılıyor...");

    // 1. ADRESLERİ ORTAM DEĞİŞKENLERİNDEN AL
    let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let qdrant_url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    let intel_url = std::env::var("INTELLIGENCE_URL").unwrap_or_else(|_| "http://localhost:50051".to_string());

    // 2. BAĞLANTILARI KUR
    let nats_client = async_nats::connect(&nats_url).await
        .context("NATS bağlantı hatası")?;

    let qdrant_client = Qdrant::from_url(&qdrant_url).build()
        .context("Qdrant bağlantı hatası")?;
    let collection_name = "market_states";

    let mut intel_client = SentimentAnalyzerClient::connect(intel_url).await
        .context("Intelligence servisine bağlanılamadı")?;

    // 3. ABONELİK VE DÖNGÜ
    let mut subscriber = nats_client.subscribe("market.trade.binance.BTCUSDT").await?;
    let mut windows: HashMap<String, WindowStats> = HashMap::new();
    let mut ticker = time::interval(Duration::from_secs(1));

    info!("⚡ VQ-Capital Zeka Katmanı AKTİF. Analiz başlıyor...");

    loop {
        tokio::select! {
            Some(message) = subscriber.next() => {
                if let Ok(trade) = AggTrade::decode(message.payload) {
                    let stats = windows.entry(trade.symbol.clone()).or_insert(WindowStats {
                        first_price: trade.price, last_price: trade.price, buy_volume: 0.0, sell_volume: 0.0, trade_count: 0,
                    });
                    stats.last_price = trade.price;
                    stats.trade_count += 1;
                    if trade.is_buyer_maker { stats.sell_volume += trade.quantity; } else { stats.buy_volume += trade.quantity; }
                }
            }

            _ = ticker.tick() => {
                for (symbol, stats) in windows.iter_mut() {
                    if stats.trade_count == 0 { continue; }

                    let price_velocity = (stats.last_price - stats.first_price) / stats.first_price;
                    let total_vol = stats.buy_volume + stats.sell_volume;
                    let volume_imbalance = if total_vol > 0.0 { (stats.buy_volume - stats.sell_volume) / total_vol } else { 0.0 };

                    // Haber Analizi (GPU)
                    let news_text = "BTC market trend analysis";
                    let intel_request = tonic::Request::new(SentimentRequest { text: news_text.to_string() });
                    let sentiment_score = match intel_client.analyze_text(intel_request).await {
                        Ok(resp) => resp.into_inner().score,
                        Err(_) => 0.0,
                    };

                    // Durum Vektörünü Yayınla
                    let current_state = MarketStateVector {
                        symbol: symbol.clone(),
                        window_start_time: 0, window_end_time: 0,
                        price_velocity, volume_imbalance, sentiment_score,
                        embeddings: vec![price_velocity, volume_imbalance, sentiment_score],
                    };
                    let mut state_buf = Vec::new();
                    current_state.encode(&mut state_buf)?;
                    let _ = nats_client.publish(format!("state.vector.{}", symbol), state_buf.into()).await;

                    // Qdrant'ta benzerlik ara
                    let current_vector_f32 = vec![price_velocity as f32, volume_imbalance as f32, sentiment_score as f32];
                    let mut signal = SignalType::Hold;
                    let mut confidence = 0.0;

                    let search_request = SearchPointsBuilder::new(collection_name, current_vector_f32, 5).with_payload(true);
                    if let Ok(search_result) = qdrant_client.search_points(search_request).await {
                        let points = search_result.result;
                        if !points.is_empty() {
                            let avg_past_velocity: f64 = points.iter()
                                .filter_map(|p| p.payload.get("velocity"))
                                .filter_map(|v| v.as_double())
                                .sum::<f64>() / points.len() as f64;

                            confidence = (avg_past_velocity.abs() * 10000.0).min(1.0);
                            if avg_past_velocity > 0.00001 { signal = SignalType::Buy; }
                            else if avg_past_velocity < -0.00001 { signal = SignalType::Sell; }
                        }
                    }

                    // Sinyali Ateşle
                    let trade_signal = TradeSignal {
                        symbol: symbol.clone(),
                        r#type: signal.into(),
                        confidence_score: confidence,
                        recommended_leverage: 1.0,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        reason: format!("V:{:.6} I:{:.2} S:{:.2}", price_velocity, volume_imbalance, sentiment_score),
                    };
                    let mut sig_buf = Vec::new();
                    trade_signal.encode(&mut sig_buf)?;
                    let _ = nats_client.publish(format!("signal.trade.{}", symbol), sig_buf.into()).await;

                    if signal != SignalType::Hold {
                        info!("🎯 [SİNYAL] {} -> {:?} (Conf: {:.2})", symbol, signal, confidence);
                    }
                    
                    // Reset
                    stats.trade_count = 0; stats.first_price = stats.last_price; stats.buy_volume = 0.0; stats.sell_volume = 0.0;
                }
            }
        }
    }
}