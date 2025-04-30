# Baxus Whisky Goggles - BAXATHON Edition! 🥃👓

Welcome to the Whisky Goggles project for the BAXATHON! Ever stared at a wall of whisky, wondering which gem you've found? Worry no more! This project uses the power of AI to identify whisky bottles from images, helping you match them against the BAXUS database.

Meet BOB, our AI agent sporting the Whisky Goggles! He's the brains (and eyes) behind the operation.

![BOB wearing Whisky Goggles](https://raw.githubusercontent.com/LoreviQ/Baxus-Honey-Barrel/main/assets/bobWG.png)

This application combines computer vision and OCR to identify whisky bottles, aiming to fulfill the hackathon bounty requirements.

## Hackathon Highlights & Features ✨

- **Dual Identification Power:** We're not just relying on one method! Whisky Goggles combines:
  - **Image Classification:** Using a custom-trained EfficientNet model (`whiskey_goggles.pth`) to recognize the bottle's visual appearance.
  - **OCR Text Extraction:** Reading the label text using Tesseract OCR for extra confirmation.
- **Custom Trained Model:** We've trained our own `whiskey_goggles_efficientnet_b0_best.pth` model on the BAXUS dataset, achieving **~85% accuracy** on image classification alone! This model is included in the `models/` directory.
- **Simple RESTful API:** Easily integrate Whisky Goggles! Send an image to the Flask API, and get back the identification results. Perfect for plugging into other applications.
- **Image Preprocessing & Enhancement:** Includes steps for cleaning up images before analysis.
- **(Optional) Upscaling:** Support for Real-ESRGAN to improve low-resolution images.
- **BOB Integration (Cross-Track Fun!):** BOB, from the AI Agent track, "wears" the goggles and performs the classification. You can interact with the Whisky Goggles via:
  - The BAXATHON submission website: [baxathon.oliver.tj/whiskeygoggles](https://baxathon.oliver.tj/whiskeygoggles)
  - The BAXATHON Chrome Extension (requires the API to be running locally): [baxathon.oliver.tj/honeybarrel](https://baxathon.oliver.tj/honeybarrel)

## Prerequisites

- Python 3.12 or higher
- Tesseract OCR engine (See installation below)
- CUDA-capable GPU (Optional, but recommended for faster processing)

## Installation

1.  **Clone the repo & Set up Python environment:**

    ```bash
    # Clone this repository first if you haven't already
    git clone <your-repo-url>
    cd Baxus-Whisky-Goggles
    # Set up your preferred Python 3.12+ environment
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _(Note: This might take a moment, especially for the deep learning libraries)_

3.  **Install Tesseract OCR:**

    ```bash
    # Ubuntu/Debian
    sudo apt-get update && sudo apt-get install -y tesseract-ocr

    # macOS
    brew install tesseract

    # Windows
    # Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
    # Make sure tesseract is added to your system's PATH
    ```

4.  **Download Models:** Ensure the necessary model files (`whiskey_goggles.pth` and `RealESRGAN_x4plus.pth`) are present in the `models/` directory.

## Usage - Let BOB Scan!

1.  **Start the Flask API server:**

    ```bash
    python src/main.py [--verbose] [--host 0.0.0.0] [--port 5000]
    ```

    - `--verbose`: Enable detailed logging.
    - `--host 0.0.0.0`: Makes the API accessible from other devices on your network (useful for testing integrations). Default is `127.0.0.1`.
    - `--port`: Port number. Default is `5000`.

2.  **Send a POST request to the `/predict` endpoint:**

    Use `curl`, Postman, or any HTTP client. Send a `multipart/form-data` request with the image file attached to the `image` field.

    ```bash
    curl -X POST -F "image=@/path/to/your/whisky_image.jpg" http://localhost:5000/predict
    ```

    _(Replace `/path/to/your/whisky_image.jpg` with the actual image path)_

    Or, use the BAXATHON website or Chrome Extension which interact with this API!

## Development

- **Run tests:**
  ```bash
  pytest
  ```

### Interested in Training Your Own Model? 🤔

Want to experiment or train the model on your own dataset? Here's how:

1.  **Prepare Your Data:** Organize your training images in a directory structure similar to `data/training_images/`, where each subdirectory is a class label.
2.  **Update Paths in `src/train.py`:**
    - Set `dataset_path` to the location of your training images.
    - Set `pickle_path` to where you want the label mapping data (`mapping_data.pkl`) to be saved.
3.  **Train the Model:**
    ```bash
    python src/train.py
    ```
    This will train the model using the data specified in `dataset_path` and save the best model checkpoints in the `models/` directory (named like `whiskey_goggles_efficientnet_b0_best_epoch<N>.pth`). It will also save the label mapping to `pickle_path`.
4.  **Test Your Trained Model:**
    - **Validation:** To evaluate your model's performance on the validation set (part of your `dataset_path`), update `model_path` in `src/train.py` to point to your desired trained model checkpoint (e.g., `models/whiskey_goggles_efficientnet_b0_best_epoch<N>.pth`) and run:
      ```bash
      python src/train.py --validate
      ```
    - **Prediction:** To test prediction on a specific image, update `model_path` and `predict_path` (the directory containing test images) in `src/train.py`. Then run:
      ```bash
      # Note: You might need to modify the hardcoded filename in train.py's predict function
      python src/train.py --predict
      ```

## Project Structure

```
├── data/               # Dataset files, processed images, etc.
├── models/             # Trained models (EfficientNet, RealESRGAN)
├── results/            # Experiment results, logs
├── scripts/            # Utility scripts (e.g., data processing)
└── src/                # Main source code
    ├── api/          # Flask API endpoints (if separated)
    ├── dataset/      # Data loading/handling
    ├── image/        # Image classification logic
    ├── ocr/          # OCR processing logic
    └── utils/        # Shared utilities
```

## Hackathon Deliverables Checklist

- [x] Working label scanning and bottle identification system (via API)
- [x] Code repository (You're looking at it!)
- [x] Implementation details and instructions (This README)
