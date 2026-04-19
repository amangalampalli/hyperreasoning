#!/usr/bin/env bash

set -e

MODEL_PATH="llama-cpp/gemma-4-E2B-it-Q3_K_M.gguf"
SERVER_BIN="./llama-cpp/libs/llama-server"
PORT=8080
CTX=8192

echo "Starting llama-server on port $PORT..."

$SERVER_BIN \
  -m $MODEL_PATH \
  -ngl 999 \
  -c $CTX \
  --port $PORT \
  --device MTL0 \
  --chat-template-kwargs '{"enable_thinking":false}'
