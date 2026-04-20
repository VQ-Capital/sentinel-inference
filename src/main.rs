// ========== DOSYA: sentinel-inference/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use prost::Message;
use std::collections::HashMap;
use tokio::time::{self, Duration};
use tracing::{info, error};

pub mod sentinel_market {
    include!(concat!(env!("OUT_DIR"), "/sentinel.market.rs"));
}
use sentinel_market::{AggTrade, MarketStateVector};

// Saniyelik verileri toplamak için geçici depo
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
    info!("🧠 Sentinel-Inference (Aggregator) başlatılıyor...");

    let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let nats_client = async_nats::connect(&nats_url).await.context("NATS bağlantı hatası")?;

    // NATS'tan ham trade verilerini dinle
    let mut subscriber = nats_client.subscribe("market.trade.binance.BTCUSDT").await?;
    
    // Sembol bazlı pencere istatistikleri
    let mut windows: HashMap<String, WindowStats> = HashMap::new();

    // 1 saniyelik periyotlarla vektör üretimi için interval
    let mut ticker = time::interval(Duration::from_secs(1));

    info!("📊 BTCUSDT trade akışı saniyelik pencerelerle işleniyor...");

    loop {
        tokio::select! {
            // A. NATS'tan yeni bir işlem geldiğinde
            Some(message) = subscriber.next() => {
                if let Ok(trade) = AggTrade::decode(message.payload) {
                    let stats = windows.entry(trade.symbol.clone()).or_insert(WindowStats {
                        first_price: trade.price,
                        last_price: trade.price,
                        buy_volume: 0.0,
                        sell_volume: 0.0,
                        trade_count: 0,
                    });

                    stats.last_price = trade.price;
                    stats.trade_count += 1;
                    
                    // Binance is_buyer_maker=true ise bu bir Taker Sell işlemidir.
                    if trade.is_buyer_maker {
                        stats.sell_volume += trade.quantity;
                    } else {
                        stats.buy_volume += trade.quantity;
                    }
                }
            }

            // B. 1 saniye dolduğunda (Vektörü hesapla ve NATS'a bas)
            _ = ticker.tick() => {
                for (symbol, stats) in windows.iter_mut() {
                    if stats.trade_count == 0 { continue; }

                    // 1. Fiyat Değişim Hızı (Velocity)
                    let price_velocity = (stats.last_price - stats.first_price) / stats.first_price;
                    
                    // 2. Alım/Satım Dengesi (Imbalance)
                    let total_vol = stats.buy_volume + stats.sell_volume;
                    let volume_imbalance = if total_vol > 0.0 {
                        (stats.buy_volume - stats.sell_volume) / total_vol
                    } else { 0.0 };

                    // 3. Vektörü oluştur (Protobuf)
                    let state_vector = MarketStateVector {
                        symbol: symbol.clone(),
                        window_start_time: 0, // Geliştirilebilir
                        window_end_time: 0,
                        price_velocity,
                        volume_imbalance,
                        sentiment_score: 0.0, // Burası C++ LLM'den gelecek
                        embeddings: vec![price_velocity, volume_imbalance], // Qdrant araması için
                    };

                    // 4. Vektörü NATS'a bas (state.vector.BTCUSDT)
                    let mut buf = Vec::new();
                    state_vector.encode(&mut buf)?;
                    let subject = format!("state.vector.{}", symbol);
                    
                    if let Err(e) = nats_client.publish(subject, buf.into()).await {
                        error!("Vektör Publish Hatası: {:?}", e);
                    } else {
                        info!("🚀 [VEKTÖR] {} | Velocity: {:.6} | Imbalance: {:.2}", 
                            symbol, price_velocity, volume_imbalance);
                    }

                    // Pencereyi sıfırla (Bir sonraki saniye için)
                    stats.trade_count = 0;
                    stats.first_price = stats.last_price;
                    stats.buy_volume = 0.0;
                    stats.sell_volume = 0.0;
                }
            }
        }
    }
}