FROM python:3.7-slim-buster

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install system-level dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Downgrade protobuf to a compatible version for TensorFlow 1.15
RUN pip install protobuf==3.20.3

# Install Python dependencies (including tf-slim)
RUN pip install tensorflow==1.15 \
    numpy \
    Keras \
    matplotlib \
    scikit-image \
    tf-slim  # <-- ADD THIS LINE to install tf-slim

# Set _NUM_CLASSES in embryo.py to 2
RUN sed -i 's/_NUM_CLASSES = 1000/_NUM_CLASSES = 2/g' scripts/slim/datasets/embryo.py

# Grant execute permission to the shell script
RUN chmod +x scripts/slim/run/load_inception_v1.sh

# Train the Inception-V1 model
RUN cd scripts/slim && ./run/load_inception_v1.sh && cd ../..

# Test the trained model
RUN cd scripts/slim && python predict.py v1 ../result/ ../../Images/test output.txt 2 && cd ../..

# Measure accuracy
RUN cd useful && python acc.py && cd ..

# (Optional) Keep container running after completion
# CMD ["/bin/bash"]