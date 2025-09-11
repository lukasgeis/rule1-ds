FROM debian:stable-slim

## Install system packages

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-seaborn \
    python3-pandas \
    python3-numpy \
    python3-tqdm \
    ca-certificates \    
    libssl-dev \
    git curl tree \
    cmake g++ gcc \
    libbz2-1.0 libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

## Building Girgs
RUN mkdir /app && \
    cd /app && \
    git clone --depth 1 https://github.com/chistopher/girgs.git && \
    cd girgs && \
    cmake -DCMAKE_BUILD_TYPE=Release -B build && \
    cmake --build build --parallel -t genhrg

## Install Rust Compiler, Build and Clean up
## Copy app sources and setup user
COPY Cargo.toml Cargo.lock  /app/
COPY src /app/src/
COPY stream-bitset /app/stream-bitset

ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal && \
    cd /app && \
    cargo build --release && \
    mv target/release/rule1 . && \
    rm -rf target && \ 
    rm -rf ~/.cargo

COPY scripts /app/scripts/

CMD [ "bash" ]