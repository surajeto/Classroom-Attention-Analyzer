FROM python:3.11-slim

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO models during build to cache them in the Docker image
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"

# Copy application files
COPY . .

# Expose port (default is 7860 for Hugging Face Spaces)
ENV PORT=7860
EXPOSE 7860

# Run uvicorn server via python main.py
CMD ["python", "main.py"]
