#!/usr/bin/env bash

echo "🔧 Updating system packages..."
apt-get update -y

echo "📦 Installing Tesseract OCR..."
apt-get install -y tesseract-ocr

echo "✅ Tesseract OCR installation complete!"
