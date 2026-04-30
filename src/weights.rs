// ========== DOSYA: sentinel-inference/src/weights.rs ==========
// UYARI: Bu dosya sentinel-optimizer (Genetik Algoritma) tarafindan otomatik olarak degistirilir.
pub fn get_dna_weights() -> Vec<f32> {
    let weights_data = vec![
        //  HOLD,    BUY,    SELL
        0.2014, 0.3419, -0.3875, // F0: Price Velocity (Z-Score)
        -0.0571, 0.6251, -0.4000, // F1: Orderbook Imbalance
        -0.2874, -0.1460, -0.4871, // F2: Neural Sentiment
        -0.3496, -0.3786, -0.2330, // F3: Chain Urgency
        -0.2739, -0.3713, -0.0890, // F4: RSI
        0.2163, -0.1000, -0.2741, // F5: Volatility
        0.0000, 0.2000, -0.2566, // F6: Taker Ratio
        -0.1753, 0.4122, -0.3247, // F7: Intensity (Tick count)
        0.2864, -0.3848, 0.4565, // F8: Position in Range
        0.1565, 0.2834, -0.0986, // F9: Orderbook Depth
        -0.0532, 0.4870, 0.0270, // F10: Time Sine (Intraday)
        0.0000, -0.1838, -0.0717, // F11: Last Close Price
    ];
    weights_data
}
