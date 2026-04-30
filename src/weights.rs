// ========== DOSYA: sentinel-inference/src/weights.rs ==========
// UYARI: Bu dosya sentinel-optimizer (Genetik Algoritma) tarafindan otomatik olarak degistirilir.
pub fn get_dna_weights() -> Vec<f32> {
    let weights_data = vec![
        //  HOLD,    BUY,    SELL
         0.2194,  0.0126,  -0.3038, // F0: Price Velocity (Z-Score)
         0.0568,  -0.2281,  -0.0222, // F1: Orderbook Imbalance
         -0.0645,  -0.2583,  0.0000, // F2: Neural Sentiment
         0.0000,  -0.2321,  0.0652, // F3: Chain Urgency
         -0.2541,  0.3632,  0.2662, // F4: RSI
         0.0000,  0.1778,  -0.4357, // F5: Volatility
         0.0055,  0.1160,  0.0000, // F6: Taker Ratio
         -0.0909,  0.0000,  0.0000, // F7: Intensity (Tick count)
         -0.2875,  -0.1922,  -0.0842, // F8: Position in Range
         0.0000,  0.0000,  0.0000, // F9: Orderbook Depth
         0.3208,  0.0234,  0.0000, // F10: Time Sine (Intraday)
         -0.0541,  0.0000,  -0.1543, // F11: Last Close Price
    ];
    weights_data
}
