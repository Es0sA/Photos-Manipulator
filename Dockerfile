FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake in model weights at build time so the container doesn't stall on the
# first real request waiting for a download.
RUN mkdir -p /app/models \
    && wget -q -O /app/models/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    && wget -q -O /app/models/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
    && wget -q -O /app/models/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth \
    && wget -q -O /app/models/colorization_deploy_v2.prototxt \
        https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt \
    && wget -q -O /app/models/pts_in_hull.npy \
        https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy \
    && wget -q -O /app/models/colorization_release_v2.caffemodel \
        http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel

COPY bot.py photo_ops.py ./

# Pre-download facexlib's face detection/parsing weights (used by GFPGAN)
# so the first real request doesn't stall on it.
RUN python3 -c "from photo_ops import build_gfpganer; build_gfpganer('/app/models')"

CMD ["python", "bot.py"]
