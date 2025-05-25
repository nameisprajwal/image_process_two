# Retail Shelf Image Processor

This application processes retail shelf images to detect products, count items, and extract text information using GPU-accelerated computer vision.

## Requirements

- Python 3.8+
- CUDA 11.7
- NVIDIA GPU (Tesla T4 or compatible)
- Ubuntu Linux

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python retail_processor.py --image path/to/your/image.jpg
```

The script will:
1. Detect products using YOLOv8
2. Perform OCR on detected regions
3. Count product instances
4. Save annotated image and product data as JSON

## Output

- Annotated image with bounding boxes saved as `output.jpg`
- Product data saved as `results.json`
- Optional: Data stored in local SQLite database

## Features

- CUDA-accelerated inference
- Product detection and counting
- Text recognition (OCR)
- Local processing (no cloud services)
- JSON output format 