// ========== DOSYA: sentinel-inference/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use prost::bytes::BytesMut; // 🔥 CERRAHİ 1: Cargo.toml değiştirmeden Prost üzerinden import
use prost::Message;
use qdrant_client::qdrant::{Condition, Filter, Range, SearchPointsBuilder};
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};
use qdrant_client::Qdrant;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout, Duration};
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
use sentinel_protos::intelligence::SemanticVector;
use sentinel_protos::market::{AggTrade, ChainUrgencyEvent, MarketStateVector, OrderbookDepth};

// -----------------------------------------------------------------------------
// 📈 QUANT MATH & ROLLING WINDOW STRUCTS (HFT ARCHITECT & QUANT APPROVED)
// -----------------------------------------------------------------------------
#[derive(Clone, Default)]
struct Bucket {
    open: f64,
    close: f64,
    buy_vol: f64,
    sell_vol: f64,
    ticks: i64,
}

#[derive(Clone)]
struct RollingWindow {
    buckets: VecDeque<Bucket>,
    current_bucket: Bucket,
    current_sec: i64,
    last_signal_ms: i64, // 🔥 DevOps: Sinyal spam engelleme kalkanı
}

impl RollingWindow {
    fn new() -> Self {
        Self {
            buckets: VecDeque::with_capacity(32), // Pre-allocated
            current_bucket: Bucket::default(),
            current_sec: 0,
            last_signal_ms: 0,
        }
    }
}

#[derive(Clone)]
struct OnlineZScore {
    mean: f64,
    variance: f64,
    alpha: f64,
    initialized: bool,
}

impl OnlineZScore {
    fn new(window: usize) -> Self {
        Self {
            mean: 0.0,
            variance: 1.0,
            alpha: 2.0 / (window as f64 + 1.0),
            initialized: false,
        }
    }

    fn update(&mut self, mut value: f64, scale: f64) -> f64 {
        value *= scale;

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
        if std_dev < 1e-6 {
            0.0
        } else {
            ((value - self.mean) / std_dev).clamp(-3.0, 3.0)
        }
    }
}

struct SymbolNormalizer {
    velocity_z: OnlineZScore,
    imbalance_z: OnlineZScore,
    sentiment_z: OnlineZScore,
    urgency_z: OnlineZScore,
    vectors_collected: i32,
}

struct InferenceState {
    sentiment_cache: HashMap<String, (f64, i64)>,
    orderbook_imbalance: HashMap<String, f64>,
    chain_urgency: HashMap<String, f64>,
    normalizers: HashMap<String, SymbolNormalizer>,
}

// -----------------------------------------------------------------------------
// 🧠 MAIN INFERENCE ENGINE
// -----------------------------------------------------------------------------
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    info!(
        "📡 Service: {} | Version: {}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());

    let _window_size_sec: i64 = std::env::var("WINDOW_SIZE_SEC")
        .unwrap_or_else(|_| "30".to_string())
        .parse()
        .unwrap_or(30);

    let min_ticks: i64 = std::env::var("MIN_TICKS")
        .unwrap_or_else(|_| "25".to_string())
        .parse()?;

    let warmup_required: i32 = std::env::var("WARMUP_VECTORS")
        .unwrap_or_else(|_| "5".to_string())
        .parse()?;

    let similarity_threshold: f32 = std::env::var("SIMILARITY_THRESHOLD")
        .unwrap_or_else(|_| "0.950".to_string())
        .parse()?;

    let blindspot_ms: i64 = std::env::var("BLINDSPOT_SEC")
        .unwrap_or_else(|_| "900".to_string())
        .parse::<i64>()?
        * 1000;

    let z_scale: f64 = std::env::var("Z_SCALE")
        .unwrap_or_else(|_| "10000.0".to_string())
        .parse()?;

    let news_decay_ms: i64 = std::env::var("NEWS_DECAY_MS")
        .unwrap_or_else(|_| "14400000".to_string())
        .parse()?;

    let vel_buy_thresh: f64 = std::env::var("VELOCITY_BUY_THRESHOLD")
        .unwrap_or_else(|_| "0.8".to_string())
        .parse()?;

    let vel_sell_thresh: f64 = std::env::var("VELOCITY_SELL_THRESHOLD")
        .unwrap_or_else(|_| "-0.8".to_string())
        .parse()?;

    let search_timeout_ms: u64 = std::env::var("SEARCH_TIMEOUT_MS")
        .unwrap_or_else(|_| "25".to_string())
        .parse()?;

    info!(
        "⚙️ Params -> Z_SCALE: {}, BUY_TH: {}, SELL_TH: {}",
        z_scale, vel_buy_thresh, vel_sell_thresh
    );

    let nats_client = async_nats::connect(&nats_url).await.context("NATS Error")?;

    let mut qdrant_opt = None;
    for _ in 0..10 {
        if let Ok(client) = Qdrant::from_url(&qdrant_url).build() {
            if client.health_check().await.is_ok() {
                qdrant_opt = Some(client);
                break;
            }
        }
        sleep(Duration::from_secs(2)).await;
    }

    let q_client = qdrant_opt.context("Qdrant Offline")?;

    loop {
        match q_client.collection_exists("market_states").await {
            Ok(exists) => {
                if !exists {
                    match q_client
                        .create_collection(
                            CreateCollectionBuilder::new("market_states")
                                .vectors_config(VectorParamsBuilder::new(4, Distance::Cosine)),
                        )
                        .await
                    {
                        Ok(_) => info!("✅ Qdrant 4D Vector Space Created."),
                        Err(e) => {
                            if !e.to_string().contains("already exists") {
                                warn!("⚠️ Collection creation failed: {}", e);
                                sleep(Duration::from_secs(2)).await;
                                continue;
                            }
                        }
                    }
                }
                break;
            }
            Err(e) => {
                warn!("⏳ Qdrant engine warming up ({}), retrying...", e);
                sleep(Duration::from_secs(2)).await;
            }
        }
    }

    let qdrant = Arc::new(q_client);

    let state = Arc::new(RwLock::new(InferenceState {
        sentiment_cache: HashMap::new(),
        orderbook_imbalance: HashMap::new(),
        chain_urgency: HashMap::new(),
        normalizers: HashMap::new(),
    }));

    let nats_ob = nats_client.clone();
    let state_ob = state.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_ob.subscribe("market.orderbook.>").await {
            while let Some(msg) = sub.next().await {
                if let Ok(depth) = OrderbookDepth::decode(msg.payload) {
                    let bids_vol: f64 = depth.bids.iter().map(|b| b.price * b.quantity).sum();
                    let asks_vol: f64 = depth.asks.iter().map(|a| a.price * a.quantity).sum();
                    let imb = if (bids_vol + asks_vol) > 0.0 {
                        (bids_vol - asks_vol) / (bids_vol + asks_vol)
                    } else {
                        0.0
                    };
                    state_ob
                        .write()
                        .await
                        .orderbook_imbalance
                        .insert(depth.symbol.to_uppercase(), imb);
                }
            }
        }
    });

    let nats_int = nats_client.clone();
    let state_int = state.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_int.subscribe("intelligence.news.vector").await {
            while let Some(msg) = sub.next().await {
                if let Ok(vec) = SemanticVector::decode(msg.payload) {
                    let symbol = vec.symbol.to_uppercase();
                    state_int
                        .write()
                        .await
                        .sentiment_cache
                        .insert(symbol.clone(), (vec.sentiment_score, vec.timestamp));
                }
            }
        }
    });

    let nats_chain = nats_client.clone();
    let state_chain = state.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_chain.subscribe("chain.urgency.>").await {
            while let Some(msg) = sub.next().await {
                if let Ok(event) = ChainUrgencyEvent::decode(msg.payload) {
                    state_chain
                        .write()
                        .await
                        .chain_urgency
                        .insert(event.symbol.to_uppercase(), event.urgency_score);
                }
            }
        }
    });

    let mut trade_sub = nats_client.subscribe("market.trade.>").await?;
    let mut rolling_windows: HashMap<String, RollingWindow> = HashMap::new();

    // 🔥 1 saniyelik pencereler üreterek 30 saniyelik kayan ortalamaları hesaplayan ana döngü
    while let Some(msg) = trade_sub.next().await {
        if let Ok(trade) = AggTrade::decode(msg.payload) {
            let symbol = trade.symbol.to_uppercase();
            let current_sec = trade.timestamp / 1000;

            let window = rolling_windows
                .entry(symbol.clone())
                .or_insert(RollingWindow::new());

            if window.current_sec == 0 {
                window.current_sec = current_sec;
                window.current_bucket.open = trade.price;
            }

            if current_sec > window.current_sec {
                // Saniye değişti, kovayı tarihe göm
                window.buckets.push_back(window.current_bucket.clone());
                if window.buckets.len() > 30 {
                    window.buckets.pop_front();
                }

                // Yeni kova
                window.current_sec = current_sec;
                window.current_bucket = Bucket::default();
                window.current_bucket.open = trade.price;

                // -----------------------------------------------------------------
                // YENİ ROLLING WINDOW (1 FPS) TETİKLEMESİ BURADA BAŞLAR
                // -----------------------------------------------------------------
                if window.buckets.len() >= 25 {
                    let first_open = window.buckets.front().unwrap().open;
                    let last_close = window.buckets.back().unwrap().close;
                    let total_buy: f64 = window.buckets.iter().map(|b| b.buy_vol).sum();
                    let total_sell: f64 = window.buckets.iter().map(|b| b.sell_vol).sum();
                    let total_ticks: i64 = window.buckets.iter().map(|b| b.ticks).sum();

                    if total_ticks >= min_ticks {
                        let velocity = if first_open > 0.0 {
                            (last_close - first_open) / first_open
                        } else {
                            0.0
                        };

                        let trade_imb = if (total_buy + total_sell) > 0.0 {
                            (total_buy - total_sell) / (total_buy + total_sell)
                        } else {
                            0.0
                        };

                        let mut st = state.write().await;
                        let ob_imb = *st.orderbook_imbalance.get(&symbol).unwrap_or(&0.0);
                        let (sent, sent_ts) = *st.sentiment_cache.get(&symbol).unwrap_or(&(0.0, 0));

                        let sentiment = if sent_ts == 0 {
                            0.0
                        } else {
                            let time_diff = trade.timestamp - sent_ts;
                            if time_diff < news_decay_ms && time_diff >= 0 {
                                sent * (1.0 - (time_diff as f64 / news_decay_ms as f64))
                            } else {
                                0.0
                            }
                        };

                        let urgency = *st.chain_urgency.get(&symbol).unwrap_or(&0.0);

                        let norm = st.normalizers.entry(symbol.clone()).or_insert_with(|| {
                            SymbolNormalizer {
                                velocity_z: OnlineZScore::new(1000), // Her saniye güncellendiği için 1000 saniye EMA
                                imbalance_z: OnlineZScore::new(1000),
                                sentiment_z: OnlineZScore::new(1000),
                                urgency_z: OnlineZScore::new(1000),
                                vectors_collected: 0,
                            }
                        });

                        norm.vectors_collected += 1;

                        let z_vel = norm.velocity_z.update(velocity, z_scale);
                        let z_imb = norm
                            .imbalance_z
                            .update((trade_imb * 0.4) + (ob_imb * 0.6), 1.0);
                        let z_sent = norm.sentiment_z.update(sentiment, 1.0);
                        let z_urg = norm.urgency_z.update(urgency, 1.0);

                        let fusion_vector =
                            vec![z_vel as f32, z_imb as f32, z_sent as f32, z_urg as f32];

                        // NATS yayınlarını saniyede bir yapıyoruz
                        let state_msg = MarketStateVector {
                            symbol: symbol.clone(),
                            window_start_time: trade.timestamp - 30_000,
                            window_end_time: trade.timestamp,
                            price_velocity: z_vel,
                            volume_imbalance: z_imb,
                            sentiment_score: z_sent,
                            chain_urgency: z_urg,
                            embeddings: fusion_vector.iter().map(|&x| x as f64).collect(),
                            map_x: z_vel as f32,
                            map_y: z_imb as f32,
                            is_gold: false,
                        };

                        let mut buf = BytesMut::with_capacity(512);
                        if state_msg.encode(&mut buf).is_ok() {
                            let _ = nats_client
                                .publish(format!("state.vector.{}", symbol), buf.into())
                                .await;
                        }

                        // 🔥 Qdrant Arama ve Sinyal Ateşleme
                        if norm.vectors_collected >= warmup_required {
                            // ⏱️ Debounce Check: Son sinyalden beri 15 sn geçti mi?
                            if trade.timestamp - window.last_signal_ms >= 15_000 {
                                let q_clone = qdrant.clone();
                                let nats_pub = nats_client.clone();
                                let sym_clone = symbol.clone();
                                let ts_clone = trade.timestamp;
                                let v_clone = fusion_vector.clone();

                                // Sinyalin spamlamaması için vakti hemen kaydediyoruz
                                window.last_signal_ms = trade.timestamp;

                                tokio::spawn(async move {
                                    let filter = Filter::all(vec![
                                        Condition::matches("symbol", sym_clone.clone()),
                                        Condition::range(
                                            "timestamp",
                                            Range {
                                                lt: Some((ts_clone - blindspot_ms) as f64),
                                                ..Default::default()
                                            },
                                        ),
                                    ]);

                                    let search_future = q_clone.search_points(
                                        SearchPointsBuilder::new("market_states", v_clone, 3)
                                            .filter(filter)
                                            .with_payload(true),
                                    );

                                    if let Ok(Ok(res)) = timeout(
                                        Duration::from_millis(search_timeout_ms),
                                        search_future,
                                    )
                                    .await
                                    {
                                        if let Some(best) = res.result.iter().find(|r| {
                                            r.score >= similarity_threshold && r.score <= 1.01
                                        }) {
                                            let mut signal = TradeSignal {
                                                symbol: sym_clone.clone(),
                                                r#type: SignalType::Hold as i32,
                                                confidence_score: best.score as f64,
                                                recommended_leverage: 1.0,
                                                timestamp: ts_clone,
                                                reason: format!(
                                                    "V4 OMNISCIENCE Match: {:.3}",
                                                    best.score
                                                ),
                                            };

                                            if z_vel > vel_buy_thresh {
                                                signal.r#type = SignalType::Buy as i32;
                                            } else if z_vel < vel_sell_thresh {
                                                signal.r#type = SignalType::Sell as i32;
                                            }

                                            if signal.r#type != SignalType::Hold as i32 {
                                                let mut buf = BytesMut::with_capacity(256);
                                                if signal.encode(&mut buf).is_ok() {
                                                    let _ = nats_pub
                                                        .publish(
                                                            format!("signal.trade.{}", sym_clone),
                                                            buf.into(),
                                                        )
                                                        .await;
                                                }
                                            }
                                        }
                                    }
                                });
                            }
                        }
                    }
                }
            }

            // Kovanın içine veriyi ekle
            window.current_bucket.close = trade.price;
            window.current_bucket.ticks += 1;
            if trade.is_buyer_maker {
                window.current_bucket.sell_vol += trade.quantity;
            } else {
                window.current_bucket.buy_vol += trade.quantity;
            }
        }
    }
    Ok(())
}
