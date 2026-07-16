import os
from pathlib import Path


def build_gfpganer(model_dir: str):
    from gfpgan import GFPGANer

    return GFPGANer(
        model_path=os.path.join(model_dir, "GFPGANv1.4.pth"),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def build_colorizer(model_dir: str):
    import cv2
    import numpy as np

    prototxt = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
    caffemodel = os.path.join(model_dir, "colorization_release_v2.caffemodel")
    pts_path = os.path.join(model_dir, "pts_in_hull.npy")

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    pts = np.load(pts_path)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full((1, 313), 2.606, dtype="float32")]

    return net


def op_remove_background(input_path: str, job_dir: str, session) -> str:
    from rembg import remove

    with open(input_path, "rb") as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes, session=session)

    output_path = os.path.join(job_dir, "no_background.png")
    with open(output_path, "wb") as f:
        f.write(output_bytes)
    return output_path


def op_restore(input_path: str, job_dir: str, gfpganer) -> str:
    import cv2

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image")

    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)

    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    _, _, restored_img = gfpganer.enhance(sharpened, has_aligned=False, only_center_face=False, paste_back=True)

    output_path = os.path.join(job_dir, "restored.png")
    cv2.imwrite(output_path, restored_img)
    return output_path


def op_colorize(input_path: str, job_dir: str, net) -> str:
    import cv2
    import numpy as np

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image")

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    l_channel = cv2.split(resized)[0]
    l_channel -= 50

    net.setInput(cv2.dnn.blobFromImage(l_channel))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    l_full = cv2.split(lab)[0]
    colorized_lab = np.concatenate((l_full[:, :, np.newaxis], ab), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    colorized_bgr = (255 * colorized_bgr).astype("uint8")

    output_path = os.path.join(job_dir, "colorized.png")
    cv2.imwrite(output_path, colorized_bgr)
    return output_path


def op_convert(input_path: str, job_dir: str, fmt: str) -> str:
    from PIL import Image

    img = Image.open(input_path)

    if fmt == "compress":
        ext = (Path(input_path).suffix or ".jpg").lstrip(".").lower()
        if ext in ("jpg", "jpeg"):
            img = img.convert("RGB")
            output_path = os.path.join(job_dir, "compressed.jpg")
            img.save(output_path, "JPEG", quality=70, optimize=True)
        else:
            output_path = os.path.join(job_dir, f"compressed.{ext}")
            img.save(output_path, optimize=True)
        return output_path

    fmt = fmt.lower()
    if fmt == "jpg":
        img = img.convert("RGB")
        output_path = os.path.join(job_dir, "converted.jpg")
        img.save(output_path, "JPEG", quality=90)
    elif fmt == "png":
        output_path = os.path.join(job_dir, "converted.png")
        img.save(output_path, "PNG")
    elif fmt == "webp":
        output_path = os.path.join(job_dir, "converted.webp")
        img.save(output_path, "WEBP", quality=90)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return output_path
