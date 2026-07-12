import asyncio
import contextlib
import dataclasses
import logging
import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
LOCAL_API_BASE_URL = os.getenv("LOCAL_API_BASE_URL", "http://telegram-bot-api:8081")
JOBS_DIR = os.getenv("PHOTOS_JOBS_DIR", "/var/lib/telegram-bot-api/_photos_jobs")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")

# Real-ESRGAN tile size: keeps CPU/memory use bounded on large photos at the
# cost of some speed. Lower this if the host runs low on memory.
UPSCALE_TILE_SIZE = 200

START_TEXT = (
    "\U0001f5bc️ *Photos Manipulator Bot*\n\n"
    "Send me a photo and I'll show you what I can do with it: upscale, "
    "remove the background, restore an old photo, colorize a black and "
    "white photo, or convert/compress the file.\n\n"
    "For best results on upscaling, restoration, or colorization, send the "
    "image as a file (the paperclip icon) rather than a compressed photo, "
    "so I get the original quality to work with. Send /help for details."
)

HELP_TEXT = (
    "*How to use*\n"
    "1. Send a photo (as a compressed photo or, better, as a file/document "
    "for full quality).\n"
    "2. Tap the button for what you want done.\n"
    "3. I'll send the result back as a file.\n\n"
    "*What I can do*\n"
    "• Upscale 2x or 4x\n"
    "• Remove the background\n"
    "• Restore an old or damaged photo (denoise, sharpen, fix faces)\n"
    "• Colorize a black and white photo\n"
    "• Convert between JPG, PNG, and WEBP, or compress the file\n\n"
    "*Limits*\n"
    "• Only one photo is processed at a time, more queue up in order.\n"
    "• Upscaling and restoration can take a few minutes on larger images, "
    "this runs on CPU with no GPU.\n"
    "• I only remember one pending photo per chat, sending a new one "
    "before you pick an action replaces it."
)

OP_LABELS = {
    "upscale2": "Upscaling 2x",
    "upscale4": "Upscaling 4x",
    "rembg": "Removing background",
    "restore": "Restoring photo",
    "colorize": "Colorizing",
}

# Keyed by (chat_id, menu_message_id) -> the attachment waiting on a button tap.
pending_photos: dict = {}

job_queue: asyncio.Queue = asyncio.Queue()
pending_jobs = 0

# Models are loaded once at startup and reused for every job; the queue
# worker processes one job at a time so there's no concurrent-access risk.
_upsampler_x2 = None
_upsampler_x4 = None
_gfpganer = None
_rembg_session = None
_colorizer_net = None


@dataclasses.dataclass
class PhotoJob:
    status_message: object
    attachment: object
    op: str = None
    fmt: str = None


def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Upscale 2x", callback_data="op:upscale2"),
                InlineKeyboardButton("Upscale 4x", callback_data="op:upscale4"),
            ],
            [InlineKeyboardButton("Remove Background", callback_data="op:rembg")],
            [InlineKeyboardButton("Restore Old Photo", callback_data="op:restore")],
            [InlineKeyboardButton("Colorize", callback_data="op:colorize")],
            [InlineKeyboardButton("Convert / Compress", callback_data="op:convert_menu")],
        ]
    )


def _convert_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("JPG", callback_data="fmt:jpg"),
                InlineKeyboardButton("PNG", callback_data="fmt:png"),
                InlineKeyboardButton("WEBP", callback_data="fmt:webp"),
            ],
            [InlineKeyboardButton("Compress (keep format)", callback_data="fmt:compress")],
            [InlineKeyboardButton("« Back", callback_data="op:back")],
        ]
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(START_TEXT, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


def _extract_photo_attachment(update: Update):
    message = update.message
    if message.photo:
        return message.photo[-1]
    if message.document and (message.document.mime_type or "").startswith("image/"):
        return message.document
    return None


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    attachment = _extract_photo_attachment(update)
    if attachment is None:
        return

    menu_message = await message.reply_text(
        "What should I do with this image?", reply_markup=_main_menu_keyboard(), quote=True
    )
    pending_photos[(message.chat_id, menu_message.message_id)] = attachment


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data
    if data == "op:convert_menu":
        await query.edit_message_text("Pick a format or compress:", reply_markup=_convert_menu_keyboard())
        return
    if data == "op:back":
        await query.edit_message_text("What should I do with this image?", reply_markup=_main_menu_keyboard())
        return

    key = (query.message.chat_id, query.message.message_id)
    attachment = pending_photos.pop(key, None)
    if attachment is None:
        await query.edit_message_text("This request expired, send the photo again.")
        return

    op = data.split(":", 1)[1] if data.startswith("op:") else None
    fmt = data.split(":", 1)[1] if data.startswith("fmt:") else None
    await _enqueue_job(query.message, attachment, op=op, fmt=fmt)


async def _enqueue_job(status_message, attachment, op, fmt) -> None:
    global pending_jobs
    pending_jobs += 1
    if pending_jobs == 1:
        text = f"{OP_LABELS.get(op, 'Processing')}..."
    else:
        text = f"Queued, {pending_jobs - 1} ahead of this one..."
    with contextlib.suppress(Exception):
        await status_message.edit_text(text)

    await job_queue.put(PhotoJob(status_message=status_message, attachment=attachment, op=op, fmt=fmt))


def _run_operation(input_path: str, job_dir: str, op: str, fmt: str):
    from photo_ops import (
        op_colorize,
        op_convert,
        op_remove_background,
        op_restore,
        op_upscale,
    )

    if op == "upscale2":
        return op_upscale(input_path, job_dir, scale=2, upsampler=_upsampler_x2), "Upscaled 2x"
    if op == "upscale4":
        return op_upscale(input_path, job_dir, scale=4, upsampler=_upsampler_x4), "Upscaled 4x"
    if op == "rembg":
        return op_remove_background(input_path, job_dir, _rembg_session), "Background removed"
    if op == "restore":
        return op_restore(input_path, job_dir, _gfpganer), "Restored"
    if op == "colorize":
        return op_colorize(input_path, job_dir, _colorizer_net), "Colorized"
    if fmt:
        label = "Compressed" if fmt == "compress" else f"Converted to {fmt.upper()}"
        return op_convert(input_path, job_dir, fmt), label
    raise ValueError(f"Unknown operation: op={op} fmt={fmt}")


async def _process_job(job: PhotoJob) -> None:
    status_message = job.status_message
    job_dir = os.path.join(JOBS_DIR, uuid.uuid4().hex)
    os.makedirs(job_dir, exist_ok=True)
    os.chmod(job_dir, 0o755)

    try:
        tg_file = await job.attachment.get_file()
        input_path = tg_file.file_path

        output_path, caption = await asyncio.get_running_loop().run_in_executor(
            None, _run_operation, input_path, job_dir, job.op, job.fmt
        )

        os.chmod(output_path, 0o644)
        with contextlib.suppress(Exception):
            await status_message.edit_text("Done, sending result...")

        await status_message.reply_document(document=output_path, caption=caption, quote=True)
        with contextlib.suppress(Exception):
            await status_message.delete()
    except Exception:
        logger.exception("Processing failed")
        with contextlib.suppress(Exception):
            await status_message.edit_text("Sorry, I couldn't process that image.")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


async def _queue_worker() -> None:
    global pending_jobs
    while True:
        job = await job_queue.get()
        try:
            await _process_job(job)
        except Exception:
            logger.exception("Unhandled error processing photo job")
        finally:
            pending_jobs -= 1
            job_queue.task_done()


def _load_models() -> None:
    global _upsampler_x2, _upsampler_x4, _gfpganer, _rembg_session, _colorizer_net
    from photo_ops import build_colorizer, build_gfpganer, build_realesrgan
    from rembg import new_session

    logger.info("Loading upscaling models...")
    _upsampler_x2 = build_realesrgan(scale=2, model_dir=MODELS_DIR, tile=UPSCALE_TILE_SIZE)
    _upsampler_x4 = build_realesrgan(scale=4, model_dir=MODELS_DIR, tile=UPSCALE_TILE_SIZE)

    logger.info("Loading GFPGAN...")
    _gfpganer = build_gfpganer(model_dir=MODELS_DIR)

    logger.info("Loading background removal session...")
    _rembg_session = new_session("u2net")

    logger.info("Loading colorization model...")
    _colorizer_net = build_colorizer(model_dir=MODELS_DIR)

    logger.info("All models loaded.")


_queue_worker_task = None


async def _post_init(application: Application) -> None:
    global _queue_worker_task
    await asyncio.get_running_loop().run_in_executor(None, _load_models)
    _queue_worker_task = asyncio.create_task(_queue_worker(), name="photo-queue-worker")


def main() -> None:
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .base_url(f"{LOCAL_API_BASE_URL}/bot")
        .base_file_url(f"{LOCAL_API_BASE_URL}/file/bot")
        .local_mode(True)
        .connect_timeout(30)
        .read_timeout(1800)
        .write_timeout(1800)
        .pool_timeout(30)
        .post_init(_post_init)
        .build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
