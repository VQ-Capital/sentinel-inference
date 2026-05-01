// ========== DOSYA: sentinel-inference/src/weights.rs ==========
// UYARI: Bu dosya sentinel-optimizer (Genetik Algoritma) tarafindan otomatik olarak degistirilir.

pub fn get_dna_weights() -> Vec<f32> {
    vec![
        //  HOLD,    BUY,    SELL
         -0.2723,  -0.4424,  -0.1284, // F0: Price Velocity (Z-Score)
         0.0910,  0.0882,  0.1586, // F1: Orderbook Imbalance
         0.2469,  -0.4110,  -0.0607, // F2: Neural Sentiment
         -0.3450,  0.1114,  -0.1546, // F3: Chain Urgency
         -0.5371,  -0.0203,  -0.2904, // F4: RSI
         -0.2007,  -0.5304,  0.6398, // F5: Volatility
         0.2147,  -0.1845,  -0.1958, // F6: Taker Ratio
         0.2172,  0.2893,  -0.2875, // F7: Intensity (Tick count)
         -0.1990,  0.0924,  0.6935, // F8: Position in Range
         -0.0590,  -0.2403,  -0.4368, // F9: Orderbook Depth
         0.3677,  0.6311,  0.6170, // F10: Time Sine (Intraday)
         -0.0020,  0.1730,  -0.2438, // F11: Last Close Price
    ]
}

pub fn get_dna_biases() -> Vec<f32> {
    vec![
        // HOLD, BUY, SELL
        0.0410, -0.2138, -0.0431, 
    ]
}
