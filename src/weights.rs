// ========== DOSYA: sentinel-inference/src/weights.rs ==========
// UYARI: Bu dosya sentinel-optimizer (Genetik Algoritma) tarafindan otomatik olarak degistirilir.

pub fn get_dna_weights() -> Vec<f32> {
    vec![
        //  HOLD,    BUY,    SELL
         -0.1610,  0.4237,  -0.0568, // F0: Price Velocity (Z-Score)
         -0.0854,  0.4707,  -0.1167, // F1: Orderbook Imbalance
         -0.0300,  -0.3225,  -0.0034, // F2: Neural Sentiment
         0.0736,  0.2994,  -0.2242, // F3: Chain Urgency
         -0.5549,  -0.0180,  -0.2972, // F4: RSI
         -0.5075,  -0.2945,  0.4250, // F5: Volatility
         0.4262,  0.0793,  0.0497, // F6: Taker Ratio
         -0.5977,  -0.1583,  0.2635, // F7: Intensity (Tick count)
         0.3855,  0.4426,  -0.4541, // F8: Position in Range
         0.1753,  0.0827,  0.0175, // F9: Orderbook Depth
         -0.4428,  0.3700,  -0.2661, // F10: Time Sine (Intraday)
         0.2433,  -0.2559,  0.2440, // F11: Last Close Price
    ]
}

pub fn get_dna_biases() -> Vec<f32> {
    vec![
        // HOLD, BUY, SELL
        0.1088, -0.1778, 0.2359, 
    ]
}
