# Baxus Whisky Goggles

A computer vision and OCR application that identifies whiskey bottles from images.

## Features

- Image classification using EfficientNet
- OCR text extraction using Tesseract
- Image preprocessing and enhancement
- RESTful API using Flask
- Support for upscaling images using Real-ESRGAN

## Prerequisites

- Python 3.12 or higher
- Tesseract OCR engine
- CUDA-capable GPU (optional, for faster processing)

## Installation

1. Set up Python environment:

```bash
pyenv local
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

1. Start the Flask API server:

```bash
python src/main.py [--verbose] [--host HOST] [--port PORT]
```

Arguments:

- `--verbose`: Enable verbose logging
- `--host`: Host address (default: 127.0.0.1)
- `--port`: Port number (default: 5000)

2. Send a POST request to `/predict` endpoint with an image file:

```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict
```

## Development

Run tests:

```bash
pytest
```

## Project Structure

```
├── data/               # Dataset and processed files
├── models/            # Trained models
├── results/           # Training results and logs
├── scripts/           # Utility scripts
└── src/              # Source code
    ├── dataset/      # Dataset handling
    ├── image/        # Image classification
    ├── ocr/          # OCR processing
    └── utils/        # Utilities
```
