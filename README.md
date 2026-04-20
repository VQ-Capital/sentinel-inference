# 🧠 sentinel-inference (The Brain)

**Sorumluluk:** Akan canlı veriyi (NATS) ve dış haber akışlarını işleyerek, Qdrant üzerindeki geçmiş piyasa vektörleriyle karşılaştırmak.
**Akış:** NATS'tan 1 saniyelik veri penceresini okur -> Yerel C++ LLM'den duygu skoru alır -> Qdrant'ta Cosine Similarity ile arama yapar -> Çıkan sonuca göre `TradeSignal` (Protobuf) oluşturup NATS `signal.trade.*` kanalına fırlatır.
**Dil:** C++ (Performans ve LLM donanım ivmelendirmesi için) / Rust.