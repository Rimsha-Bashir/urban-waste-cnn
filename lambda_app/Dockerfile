FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies required by Pillow & PyTorch
RUN yum install -y \
        libjpeg-turbo-devel \
        zlib-devel \
        libXext \
        libSM \
        libXrender \
    && yum clean all

# Install Python dependencies
RUN pip install --no-cache-dir \
        numpy \
        Pillow

# Install CPU-only PyTorch (Lambda-compatible)
RUN pip install --no-cache-dir \
        torch \
        torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY lambda_app/app.py ${LAMBDA_TASK_ROOT}/app.py

# Copy model folder into container
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# Lambda entrypoint
CMD ["app.handler"]