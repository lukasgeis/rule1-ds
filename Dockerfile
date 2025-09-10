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
    ca-certificates \    
    libssl-dev \
    git curl tree \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

## Copy app sources and setup user
COPY Cargo.toml Cargo.lock  /app/
COPY src /app/src/
COPY scripts /app/scripts/
COPY stream-bitset /app/stream-bitset

## Drop root user permissions
ARG USER_ID=1000
ARG GROUP_ID=1000    
RUN groupadd -g ${GROUP_ID} rustuser && \
    useradd -m -u ${USER_ID} -g rustuser -s /bin/bash rustuser && \
    echo "rustuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown rustuser:rustuser -R /app && tree /app

USER rustuser
WORKDIR /home/rustuser

## Install Rust Compiler, Build and Clean up
ENV PATH="/home/rustuser/.cargo/bin:${PATH}"

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal && \
    cd /app && \
    cargo build --release && \
    mv target/release/rule1 . && \
    rm -rf target && \ 
    rm -rf ~/.cargo

CMD [ "bash" ]