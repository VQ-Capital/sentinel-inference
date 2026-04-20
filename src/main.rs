use anyhow::Result;
use futures_util::StreamExt;
use prost::Message;
use qdrant_client::Qdrant;
use std::collections::HashMap;
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
use sentinel_protos::intelligence::sentiment_analyzer_client::SentimentAnalyzerClient;
use sentinel_protos::intelligence::SentimentRequest;
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

    let nats_client = async_nats::connect(&nats_url).await?;
    let _qdrant_client = Qdrant::from_url(&qdrant_url).build()?;
    let mut intel_client = SentimentAnalyzerClient::connect(intel_url).await?;

    let mut subscriber = nats_client.subscribe("market.trade.binance.>").await?;
    let mut windows: HashMap<String, WindowStats> = HashMap::new();

    info!("🧠 Inference başlatıldı. Sinyal motoru AKTİF.");

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

                    // 💡 YENİ: Fiyat hareketine göre Yapay Zekaya gönderilecek metni oluştur!
                    let simulated_news = if price_velocity > 0.0001 {
                        "Massive breakout seen, market looks highly bullish and ready to moon!"
                    } else if price_velocity < -0.0001 {
                        "Market faces heavy selloff, bearish trend indicates a crash and dump."
                    } else {
                        "Market is ranging near support, accumulation ongoing."
                    };

                    let intel_request = tonic::Request::new(SentimentRequest {
                        text: simulated_news.into(),
                    });
                    let sentiment_score = intel_client
                        .analyze_text(intel_request)
                        .await
                        .map(|r| r.into_inner().score)
                        .unwrap_or(0.0);

                    // SİNYAL ÜRETİMİ
                    let signal = if price_velocity >= 0.0 {
                        SignalType::Buy
                    } else {
                        SignalType::Sell
                    };

                    let trade_signal = TradeSignal {
                        symbol: trade.symbol.clone(),
                        r#type: signal.into(),
                        confidence_score: 0.85,
                        recommended_leverage: 1.0,
                        timestamp: trade.timestamp,
                        reason: "HFT Trigger".into(),
                    };
                    let mut sig_buf = Vec::new();
                    trade_signal.encode(&mut sig_buf)?;
                    let _ = nats_client
                        .publish(format!("signal.trade.{}", trade.symbol), sig_buf.into())
                        .await;

                    // Vektörü Qdrant'a yollama
                    let current_state = MarketStateVector {
                        symbol: trade.symbol.clone(),
                        window_start_time: 0,
                        window_end_time: 0,
                        price_velocity,
                        volume_imbalance,
                        sentiment_score,
                        embeddings: vec![], // Embeddings Storage'da dolduruluyor
                    };
                    let mut state_buf = Vec::new();
                    current_state.encode(&mut state_buf)?;
                    let _ = nats_client
                        .publish(format!("state.vector.{}", trade.symbol), state_buf.into())
                        .await;
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
