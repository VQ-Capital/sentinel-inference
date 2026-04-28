# ========== DOSYA: sentinel-inference/Dockerfile ==========

# 1. AŞAMA: RUST DERLEYİCİSİ (PURE RUST BUILDER)
FROM rust:1.95-slim-bookworm AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    protobuf-compiler pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

# 2. AŞAMA: ÜRETİM ORTAMI (ZERO C++ DEPENDENCY & NATIVE SPEED)
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Binary'yi kopyalıyoruz
COPY --from=builder /usr/src/app/target/release/sentinel-inference .

ENV OMP_NUM_THREADS=1

CMD ["./sentinel-inference"]