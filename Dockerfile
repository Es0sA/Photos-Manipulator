FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch/torchvision in their own step, with --index-url
# (not --extra-index-url) so pip can't mix in GPU-flavored wheels from the
# default PyPI index, which caused a version conflict with basicsr's
# transitive nvidia-cublas requirement.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake in model weights at build time so the container doesn't stall on the
# first real request waiting for a download.
RUN mkdir -p /app/models \
    && wget -O /app/models/RealESRGAN_x4plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
    && wget -O /app/models/RealESRGAN_x2plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    && wget -O /app/models/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
    && wget -O /app/models/colorization_deploy_v2.prototxt \
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt" \
    && wget -O /app/models/pts_in_hull.npy \
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy" \
    && wget -O /app/models/colorization_release_v2.caffemodel \
        "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"

COPY bot.py photo_ops.py ./

# Pre-download facexlib's face detection/parsing weights (used by GFPGAN)
# and rembg's u2net model, so the first real request doesn't stall on
# either, and container restarts/recreates don't redownload them.
RUN python3 -c "from photo_ops import build_gfpganer; build_gfpganer('/app/models')" \
    && python3 -c "from rembg import new_session; new_session('u2net')"

CMD ["python", "bot.py"]
