FROM python:3.10-slim

# Install system libraries needed for OpenCV and dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run model check / download if missing
RUN python setup_models.py

EXPOSE 5000

ENV PORT=5000

CMD ["gunicorn", "--threads", "4", "-b", "0.0.0.0:5000", "app:app"]
