# Stage 1: Base image with Triton pre-installed
FROM nvcr.io/nvidia/tritonserver:24.08-py3 AS base

# Stage 2: Install Python dependencies (vLLM, ONNX Runtime)
# Note: Triton 24.08 already ships with PyTorch and CUDA 12.x pre-configured.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir vllm==0.5.4 onnxruntime-gpu

# Stage 3: Final Runtime Image
FROM base AS final

WORKDIR /opt/tritonserver

# Copy the local model repository into the container image
COPY model_repository /opt/model_repository

EXPOSE 8000 8001 8002

ENTRYPOINT ["tritonserver"]
# Use 'poll' or 'none' if you want models loaded automatically at boot; 
# keep 'explicit' if you manage model loading dynamically via REST/gRPC API.
# Corrected CMD
CMD ["--model-repository=/opt/model_repository"]