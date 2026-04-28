// ========== DOSYA: sentinel-inference/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use ort::{execution_providers::CUDAExecutionProvider, session::Session, value::Value};
use prost::bytes::BytesMut;
use prost::Message;
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};
use qdrant_client::Qdrant;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex}; // 🔥 CERRAHİ: Mutex Eklendi
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout, Duration};
use tracing::{error, info, warn};

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

#[derive(Clone, Default)]
struct Bucket {
    open: f64,
    close: f64,
    high: f64,
    low: f64,
    buy_vol: f64,
    sell_vol: f64,
    ticks: i64,
}

#[derive(Clone)]
struct RollingWindow {
    buckets: VecDeque<Bucket>,
    current_bucket: Bucket,
    current_sec: i64,
    last_signal_ms: i64,
}

impl RollingWindow {
    fn new() -> Self {
        Self {
            buckets: VecDeque::with_capacity(64),
            current_bucket: Bucket {
                high: f64::MIN,
                low: f64::MAX,
                ..Default::default()
            },
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
    z_scores: Vec<OnlineZScore>,
    vectors_collected: i32,
}

impl SymbolNormalizer {
    fn new() -> Self {
        Self {
            z_scores: vec![OnlineZScore::new(1000); 12], // 12 Boyutlu Uzay için
            vectors_collected: 0,
        }
    }
}

struct InferenceState {
    sentiment_cache: HashMap<String, (f64, i64)>,
    orderbook_imbalance: HashMap<String, f64>,
    orderbook_depth: HashMap<String, f64>,
    chain_urgency: HashMap<String, f64>,
    normalizers: HashMap<String, SymbolNormalizer>,
}

// -----------------------------------------------------------------------------
// 🧠 ONNX ML ENGINE W/ GRACEFUL DEGRADATION
// -----------------------------------------------------------------------------
// 🔥 CERRAHİ: Artık Session bir Mutex üzerinden güvenle alınıyor
fn run_onnx_inference(
    session_mutex: &Mutex<Session>,
    features: &[f32; 12],
) -> Result<(SignalType, f64)> {
    let tensor = Value::from_array(([1, 12], features.to_vec()))?;

    // Interior mutability ile thread-safe kilit alıyoruz
    let mut session = session_mutex
        .lock()
        .map_err(|e| anyhow::anyhow!("AI Mutex Poisoned: {}", e))?;

    let outputs = session.run(ort::inputs!["input" => tensor])?;
    let logits = outputs["output"].try_extract_tensor::<f32>()?;
    let slice = logits.1;

    if slice.len() >= 3 {
        let hold_prob = slice[0];
        let buy_prob = slice[1];
        let sell_prob = slice[2];

        if buy_prob > hold_prob && buy_prob > sell_prob {
            Ok((SignalType::Buy, buy_prob as f64))
        } else if sell_prob > hold_prob && sell_prob > buy_prob {
            Ok((SignalType::Sell, sell_prob as f64))
        } else {
            Ok((SignalType::Hold, hold_prob as f64))
        }
    } else {
        Ok((SignalType::Hold, 0.0))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!(
        "📡 Service: {} | Version: {} (TRUE AI / ONNX EDITION)",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );

    // Çevresel Değişkenler
    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string());
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
    let model_path = std::env::var("ONNX_MODEL_PATH")
        .unwrap_or_else(|_| "/opt/models/hft_quant_v1.onnx".to_string());
    let min_ticks: i64 = std::env::var("MIN_TICKS")
        .unwrap_or_else(|_| "25".to_string())
        .parse()?;
    let warmup_required: i32 = std::env::var("WARMUP_VECTORS")
        .unwrap_or_else(|_| "5".to_string())
        .parse()?;
    let z_scale: f64 = std::env::var("Z_SCALE")
        .unwrap_or_else(|_| "10000.0".to_string())
        .parse()?;
    let ai_timeout_ms: u64 = std::env::var("AI_TIMEOUT_MS")
        .unwrap_or_else(|_| "25".to_string())
        .parse()?;

    let nats_client = async_nats::connect(&nats_url).await.context("NATS Error")?;

    // 1. QDRANT BAZLI HAFIZA (12D Güncellemesi)
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

    if !q_client
        .collection_exists("market_states_12d")
        .await
        .unwrap_or(false)
    {
        let _ = q_client
            .create_collection(
                CreateCollectionBuilder::new("market_states_12d")
                    .vectors_config(VectorParamsBuilder::new(12, Distance::Cosine)),
            )
            .await;
    }

    // 2. ONNX YAPAY ZEKA MODELİNİN YÜKLENMESİ
    let builder_initial = Session::builder().context("ONNX Builder başlatılamadı")?;

    // 🔥 CERRAHİ: `mut builder` yapılarak mutability hatası çözüldü
    let mut builder = builder_initial
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .unwrap_or_else(|e| {
            warn!(
                "⚠️ CUDA başlatılamadı, Saf CPU MLOps moduna düşülüyor. Detay: {}",
                e
            );
            Session::builder().unwrap()
        });

    let session = builder
        .commit_from_file(&model_path)
        .context(format!("Model dosyası okunamadı: {}", model_path))?;

    // 🔥 CERRAHİ: Session Mutex içine alındı
    let session_arc = Arc::new(Mutex::new(session));
    info!("🧠 ONNX Model Yüklendi ve Hazır!");

    let state = Arc::new(RwLock::new(InferenceState {
        sentiment_cache: HashMap::new(),
        orderbook_imbalance: HashMap::new(),
        orderbook_depth: HashMap::new(),
        chain_urgency: HashMap::new(),
        normalizers: HashMap::new(),
    }));

    // ORDERBOOK LISTENER
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
                    let mut st = state_ob.write().await;
                    st.orderbook_imbalance
                        .insert(depth.symbol.to_uppercase(), imb);
                    st.orderbook_depth
                        .insert(depth.symbol.to_uppercase(), bids_vol + asks_vol);
                }
            }
        }
    });

    // SENTIMENT LISTENER
    let nats_int = nats_client.clone();
    let state_int = state.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_int.subscribe("intelligence.news.vector").await {
            while let Some(msg) = sub.next().await {
                if let Ok(vec) = SemanticVector::decode(msg.payload) {
                    state_int.write().await.sentiment_cache.insert(
                        vec.symbol.to_uppercase(),
                        (vec.sentiment_score, vec.timestamp),
                    );
                }
            }
        }
    });

    // CHAIN URGENCY LISTENER
    let nats_chain = nats_client.clone();
    let _state_chain = state.clone();
    tokio::spawn(async move {
        if let Ok(mut sub) = nats_chain.subscribe("chain.urgency.>").await {
            while let Some(_msg) = sub.next().await {
                // Placeholder for future chain parsing logic
            }
        }
    });

    let mut trade_sub = nats_client.subscribe("market.trade.>").await?;
    let mut rolling_windows: HashMap<String, RollingWindow> = HashMap::new();

    // 3. HOT-PATH: MAIN TICK PROCESSOR
    while let Some(msg) = trade_sub.next().await {
        if let Ok(trade) = AggTrade::decode(msg.payload) {
            let symbol = trade.symbol.to_uppercase();
            let current_sec = trade.timestamp / 1000;
            let window = rolling_windows
                .entry(symbol.clone())
                .or_insert_with(RollingWindow::new);

            if window.current_sec == 0 {
                window.current_sec = current_sec;
                window.current_bucket.open = trade.price;
                window.current_bucket.high = trade.price;
                window.current_bucket.low = trade.price;
            }

            if current_sec > window.current_sec {
                window.buckets.push_back(window.current_bucket.clone());
                if window.buckets.len() > 60 {
                    // Son 1 dakikayı (60 sn) hafızada tut
                    window.buckets.pop_front();
                }

                window.current_sec = current_sec;
                window.current_bucket = Bucket {
                    open: trade.price,
                    high: trade.price,
                    low: trade.price,
                    ..Default::default()
                };

                if window.buckets.len() >= 25 {
                    // Feature hesaplamaları
                    let first_open = window.buckets.front().unwrap().open;
                    let last_close = window.buckets.back().unwrap().close;

                    let mut total_buy = 0.0;
                    let mut total_sell = 0.0;
                    let mut total_ticks = 0;
                    let mut highest = f64::MIN;
                    let mut lowest = f64::MAX;
                    let mut gains = 0.0;
                    let mut losses = 0.0;

                    let mut prev_close = first_open;

                    for b in window.buckets.iter() {
                        total_buy += b.buy_vol;
                        total_sell += b.sell_vol;
                        total_ticks += b.ticks;
                        if b.high > highest {
                            highest = b.high;
                        }
                        if b.low < lowest {
                            lowest = b.low;
                        }

                        let change = b.close - prev_close;
                        if change > 0.0 {
                            gains += change;
                        } else {
                            losses += change.abs();
                        }
                        prev_close = b.close;
                    }

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
                        let volatility = if lowest > 0.0 {
                            (highest - lowest) / lowest
                        } else {
                            0.0
                        };
                        let rsi = if (gains + losses) > 0.0 {
                            100.0 - (100.0 / (1.0 + (gains / losses.max(1e-9))))
                        } else {
                            50.0
                        };
                        let taker_ratio = if (total_buy + total_sell) > 0.0 {
                            total_buy / (total_buy + total_sell)
                        } else {
                            0.5
                        };
                        let intensity = total_ticks as f64 / window.buckets.len() as f64;
                        let position_in_range = if highest > lowest {
                            (last_close - lowest) / (highest - lowest)
                        } else {
                            0.5
                        };
                        let hour = chrono::Utc::now()
                            .time()
                            .format("%H")
                            .to_string()
                            .parse::<f64>()
                            .unwrap_or(0.0);
                        let time_sin = (hour * std::f64::consts::PI / 12.0).sin();

                        let mut st = state.write().await;
                        let ob_imb = *st.orderbook_imbalance.get(&symbol).unwrap_or(&0.0);
                        let ob_depth = *st.orderbook_depth.get(&symbol).unwrap_or(&1.0);
                        let (sent, sent_ts) = *st.sentiment_cache.get(&symbol).unwrap_or(&(0.0, 0));
                        let sentiment = if sent_ts > 0 && (trade.timestamp - sent_ts) < 14400000 {
                            sent
                        } else {
                            0.0
                        };
                        let urgency = *st.chain_urgency.get(&symbol).unwrap_or(&0.0);

                        let norm = st
                            .normalizers
                            .entry(symbol.clone())
                            .or_insert_with(SymbolNormalizer::new);
                        norm.vectors_collected += 1;

                        // 12 Boyutlu Z-Score Normalizasyonu (Sırasıyla)
                        let mut features = [0.0f32; 12];
                        features[0] = norm.z_scores[0].update(velocity, z_scale) as f32;
                        features[1] =
                            norm.z_scores[1].update((trade_imb * 0.4) + (ob_imb * 0.6), 1.0) as f32;
                        features[2] = norm.z_scores[2].update(sentiment, 1.0) as f32;
                        features[3] = norm.z_scores[3].update(urgency, 1.0) as f32;
                        features[4] = norm.z_scores[4].update(rsi, 1.0) as f32;
                        features[5] = norm.z_scores[5].update(volatility, z_scale) as f32;
                        features[6] = norm.z_scores[6].update(taker_ratio, 1.0) as f32;
                        features[7] = norm.z_scores[7].update(intensity, 1.0) as f32;
                        features[8] = norm.z_scores[8].update(position_in_range, 1.0) as f32;
                        features[9] = norm.z_scores[9].update(ob_depth, 1.0) as f32;
                        features[10] = norm.z_scores[10].update(time_sin, 1.0) as f32;
                        features[11] = norm.z_scores[11].update(last_close, 1.0) as f32;

                        let state_msg = MarketStateVector {
                            symbol: symbol.clone(),
                            window_start_time: trade.timestamp - 30_000,
                            window_end_time: trade.timestamp,
                            price_velocity: features[0] as f64,
                            volume_imbalance: features[1] as f64,
                            sentiment_score: features[2] as f64,
                            chain_urgency: features[3] as f64,
                            embeddings: features.iter().map(|&x| x as f64).collect(),
                            map_x: features[0],
                            map_y: features[1],
                            is_gold: false,
                        };

                        let mut buf = BytesMut::with_capacity(512);
                        if state_msg.encode(&mut buf).is_ok() {
                            let _ = nats_client
                                .publish(format!("state.vector.{}", symbol), buf.split().freeze())
                                .await;
                        }

                        // 🔥 ONNX MODEL ÇIKARIMI (SLA TIMEOUT KORUMALI)
                        if norm.vectors_collected >= warmup_required
                            && (trade.timestamp - window.last_signal_ms >= 15_000)
                        {
                            let session_clone = session_arc.clone(); // Arc<Mutex<Session>> kopyası
                            let features_clone = features;
                            let nats_pub = nats_client.clone();
                            let sym_clone = symbol.clone();
                            let ts_clone = trade.timestamp;

                            window.last_signal_ms = trade.timestamp;

                            tokio::spawn(async move {
                                let ai_result = timeout(
                                    Duration::from_millis(ai_timeout_ms),
                                    tokio::task::spawn_blocking(move || {
                                        run_onnx_inference(&session_clone, &features_clone)
                                    }),
                                )
                                .await;

                                match ai_result {
                                    Ok(Ok(Ok((signal_type, confidence)))) => {
                                        if signal_type != SignalType::Hold && confidence > 0.65 {
                                            let signal = TradeSignal {
                                                symbol: sym_clone.clone(),
                                                r#type: signal_type as i32,
                                                confidence_score: confidence,
                                                recommended_leverage: 1.0,
                                                timestamp: ts_clone,
                                                reason: format!(
                                                    "V4 ONNX AI Inference: {:.2}% Confidence",
                                                    confidence * 100.0
                                                ),
                                            };

                                            let mut s_buf = BytesMut::with_capacity(256);
                                            if signal.encode(&mut s_buf).is_ok() {
                                                let _ = nats_pub
                                                    .publish(
                                                        format!("signal.trade.{}", sym_clone),
                                                        s_buf.split().freeze(),
                                                    )
                                                    .await;
                                                info!(
                                                    "🎯 [AI-SIGNAL] {} -> {:?} (Conf: {:.2})",
                                                    sym_clone, signal_type, confidence
                                                );
                                            }
                                        }
                                    }
                                    Ok(Ok(Err(e))) => {
                                        error!("ONNX Tensor Error: {}", e);
                                    }
                                    Ok(Err(_)) => {
                                        // Spawn blocking thread error
                                    }
                                    Err(_) => {
                                        warn!("⏳ [SLA-BREACH] AI inference exceeded {}ms! Falling back to HOLD.", ai_timeout_ms);
                                    }
                                }
                            });
                        }
                    }
                }
            }

            window.current_bucket.close = trade.price;
            if trade.price > window.current_bucket.high {
                window.current_bucket.high = trade.price;
            }
            if trade.price < window.current_bucket.low {
                window.current_bucket.low = trade.price;
            }
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
