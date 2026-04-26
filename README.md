# 🧠 sentinel-quant-fusion (Legacy: sentinel-inference)

**Domain:** Z-Score Normalization & V4 Omniscience Vector Engine
**Rol:** Sistemin Beyni (The Brain)

Bu servis, gelen çok boyutlu verileri (Fiyat İvmesi, Tahta Dengesizliği, Haber Duygusu, Zincir Aciliyeti) anlık olarak `OnlineZScore` ile normalize eder. Bu 4D vektörü Qdrant üzerinde geçmiş anılarla (Cosine Similarity) karşılaştırır ve matematiksel bir kesişim bulduğunda işlemi (TradeSignal) ateşler.

- **NATS Girdisi:** `market.trade.*`, `market.orderbook.*`, `intelligence.news.vector`, `chain.urgency.*`
- **NATS Çıktısı:** `signal.trade.*`, `state.vector.*`
- **SLA Hedefi:** < 25ms