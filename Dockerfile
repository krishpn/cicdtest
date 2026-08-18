# Stage 1: Triton Base with ONNX Runtime & Python Support
FROM nvcr.io/nvidia/tritonserver:24.08-py3 AS base

# Stage 2: Dependencies for vLLM Execution
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      vllm==0.5.4 \
      onnxruntime-gpu \
      torch --index-url https://download.pytorch.org/whl/cu121

# Stage 3: Final Runtime Image
WORKDIR /opt/tritonserver
EXPOSE 8000 8001 8002

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/opt/model_repository", "--model-control-mode=explicit"]
