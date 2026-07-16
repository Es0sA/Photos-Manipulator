# Photos Manipulator: Telegram Bot

A Telegram bot that edits photos on request: background removal, old photo
restoration, colorization, and format conversion/compression. Send a photo,
tap a button, get the result back.

## How it works

1. Send the bot a photo, as a compressed photo or, better, as a file
   (paperclip icon) for full original quality.
2. The bot replies with a menu of buttons: Remove Background, Restore Old
   Photo, Colorize, Convert/Compress.
3. Tap one. The bot processes it and sends the result back as a file
   (never as a compressed photo, so resolution and transparency survive).
4. If you send several photos, they're queued and processed one at a time
   in the order received.

## Why this bot needs its own local Bot API server

Telegram's standard (cloud) Bot API caps both downloads and uploads at
20MB. Restored images can exceed that, so this bot runs alongside a
self-hosted
[`telegram-bot-api`](https://github.com/tdlib/telegram-bot-api) server (the
same one Telegram runs, just self-hosted), which raises the limit to
2000MB. The bot container and the `telegram-bot-api` container share a
Docker volume: files are handed off by filesystem path instead of over
HTTP.

## 1. Get API credentials

Running your own Bot API server requires a Telegram api_id / api_hash pair
(separate from the bot token), tied to your personal Telegram account:

1. Go to https://my.telegram.org and log in with your phone number.
2. Open "API development tools".
3. Create an app (any name/description works) and copy the `api_id` and
   `api_hash` it gives you. Keep these secret.

## 2. Create the bot on Telegram

1. Open a chat with [@BotFather](https://t.me/BotFather).
2. Send `/newbot` and follow the prompts.
3. BotFather gives you a token (looks like `<numbers>:<35-char string>`).
   Keep it secret, anyone with it can control your bot.

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in `TELEGRAM_BOT_TOKEN`, `TELEGRAM_API_ID`, and
`TELEGRAM_API_HASH`.

## 4. Run it

```bash
docker compose up -d --build
docker compose logs -f    # watch it come online / debug
```

The first build downloads several hundred MB of model weights (GFPGAN,
the colorization model) and bakes them into the image, so restarts don't
re-download them.

## What each feature actually does

| Feature | Model / method | Notes |
|---|---|---|
| Remove Background | rembg (U2Net) | Outputs a transparent PNG. |
| Restore Old Photo | OpenCV denoise/sharpen + GFPGAN | Denoises and sharpens the full image, then specifically restores detected faces. |
| Colorize | OpenCV DNN (Zhang et al. colorization) | Classic colorization model, decent results on most black and white photos. |
| Convert / Compress | Pillow | JPG, PNG, WEBP conversion, or quality-based compression. |

## Limits to know about

- This runs on CPU only, no GPU. Restoration can take from a few seconds
  to a few minutes depending on image size.
- Only one photo is processed at a time (see `job_queue` / `_queue_worker`
  in `bot.py`), more sent in the meantime queue up instead of running in
  parallel.
- The bot only remembers one pending photo per chat, sending a new one
  before you pick an action replaces it.
