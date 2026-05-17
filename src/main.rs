// ========== DOSYA: sentinel-inference/src/main.rs ==========
use anyhow::{Context, Result};
use futures_util::StreamExt;
use prost::bytes::BytesMut;
use prost::Message;
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};
use qdrant_client::Qdrant;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout, Duration};
use tracing::{info, warn};

use sentinel_core::math::model::PureMathModel;
use sentinel_core::math::zscore::OnlineZScore;
use sentinel_core::types::SignalType as CoreSignalType;

mod config;
mod weights;

use config::AppConfig;
use weights::{get_dna_b1, get_dna_b2, get_dna_w1, get_dna_w2};

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
            z_scores: vec![OnlineZScore::new(1000); 12],
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let config = AppConfig::from_env();

    info!(
        "📡 Service: {} | Version: 6.0.0 (V13 MLP NON-LINEAR SYNC)",
        env!("CARGO_PKG_NAME")
    );

    let nats_client = async_nats::connect(&config.nats_url)
        .await
        .context("NATS Error")?;

    let mut qdrant_opt = None;
    for _ in 0..10 {
        if let Ok(client) = Qdrant::from_url(&config.qdrant_url).build() {
            if client.health_check().await.is_ok() {
                qdrant_opt = Some(client);
                break;
            }
        }
        sleep(Duration::from_secs(2)).await;
    }
    let q_client = qdrant_opt.context("Qdrant Offline")?;

    if !q_client
        .collection_exists(&config.qdrant_collection)
        .await
        .unwrap_or(false)
    {
        let _ = q_client
            .create_collection(
                CreateCollectionBuilder::new(&config.qdrant_collection)
                    .vectors_config(VectorParamsBuilder::new(12, Distance::Cosine)),
            )
            .await;
    }

    let math_model = Arc::new(PureMathModel::new(
        get_dna_w1(),
        get_dna_b1(),
        get_dna_w2(),
        get_dna_b2(),
    )?);

    let state = Arc::new(RwLock::new(InferenceState {
        sentiment_cache: HashMap::new(),
        orderbook_imbalance: HashMap::new(),
        orderbook_depth: HashMap::new(),
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
                    let mut st = state_ob.write().await;
                    st.orderbook_imbalance
                        .insert(depth.symbol.to_uppercase(), imb);
                    st.orderbook_depth
                        .insert(depth.symbol.to_uppercase(), bids_vol + asks_vol);
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
                    state_int.write().await.sentiment_cache.insert(
                        vec.symbol.to_uppercase(),
                        (vec.sentiment_score, vec.timestamp),
                    );
                }
            }
        }
    });

    let mut trade_sub = nats_client.subscribe("market.trade.>").await?;
    let mut rolling_windows: HashMap<String, RollingWindow> = HashMap::new();

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
                    let first_open = window
                        .buckets
                        .front()
                        .map(|b| b.open)
                        .unwrap_or(trade.price);
                    let last_close = window
                        .buckets
                        .back()
                        .map(|b| b.close)
                        .unwrap_or(trade.price);

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

                    if total_ticks >= config.min_ticks {
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
                        let time_sin = 0.0;

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

                        let mut features = [0.0f32; 12];
                        features[0] = norm.z_scores[0].update(velocity, config.z_scale) as f32;
                        features[1] =
                            norm.z_scores[1].update((trade_imb * 0.4) + (ob_imb * 0.6), 1.0) as f32;
                        features[2] = norm.z_scores[2].update(sentiment, 1.0) as f32;
                        features[3] = norm.z_scores[3].update(urgency, 1.0) as f32;
                        features[4] = norm.z_scores[4].update(rsi, 1.0) as f32;
                        features[5] = norm.z_scores[5].update(volatility, config.z_scale) as f32;
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

                        if norm.vectors_collected >= config.warmup_vectors {
                            let model_clone = math_model.clone();
                            let features_clone = features;
                            let nats_pub = nats_client.clone();
                            let sym_clone = symbol.clone();
                            let ts_clone = trade.timestamp;
                            let timeout_val = config.ai_timeout_ms;
                            let min_conf = config.min_confidence_score;

                            tokio::spawn(async move {
                                let ai_result = timeout(
                                    Duration::from_millis(timeout_val),
                                    tokio::task::spawn_blocking(move || {
                                        model_clone.predict(&features_clone)
                                    }),
                                )
                                .await;

                                match ai_result {
                                    Ok(Ok(Ok((core_sig_type, confidence)))) => {
                                        let pb_sig_type = match core_sig_type {
                                            CoreSignalType::Buy => SignalType::Buy,
                                            CoreSignalType::Sell => SignalType::Sell,
                                            CoreSignalType::StrongBuy => SignalType::StrongBuy,
                                            CoreSignalType::StrongSell => SignalType::StrongSell,
                                            _ => SignalType::Hold,
                                        };

                                        if pb_sig_type != SignalType::Hold && confidence > min_conf
                                        {
                                            let signal = TradeSignal {
                                                symbol: sym_clone.clone(),
                                                r#type: pb_sig_type as i32,
                                                confidence_score: confidence,
                                                recommended_leverage: 1.0,
                                                timestamp: ts_clone,
                                                reason: format!(
                                                    "V13 MLP SYNC: {:.2}%",
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
                                            }
                                        }
                                    }
                                    Ok(Ok(Err(_))) | Ok(Err(_)) => {}
                                    Err(_) => warn!("⏳ [SLA-BREACH] AI Timeout!"),
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
