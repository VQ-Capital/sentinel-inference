# ========== DOSYA: Dockerfile (İlgili her repo için) ==========
# 1. Derleme Aşaması
FROM rust:1.75-slim-bookworm AS builder

# Sistem bağımlılıkları (Protobuf derleyicisi için şart)
RUN apt-get update && apt-get install -y protobuf-compiler pkg-config libssl-dev

WORKDIR /usr/src/app
COPY . .

# Release derlemesi
RUN cargo build --release

# 2. Çalıştırma Aşaması (Tertemiz ve Küçük)
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Binaries ismini her repoya göre cargo build çıktısından kopyalar
# sentinel-ingest için: sentinel-ingest, sentinel-storage için: sentinel-storage vb.
COPY --from=builder /usr/src/app/target/release/sentinel-* .

# ÇALIŞTIRMA KOMUTU (Not: Servis adını buraya manuel yazmalısın veya docker-compose command kullanmalısın)
# Biz docker-compose içinde belirteceğiz.