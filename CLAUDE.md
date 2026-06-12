# CLAUDE.md - BackgroundRemover.app

## Overview

Dual-purpose repository: (1) the `backgroundremover` pip package (CLI tool for AI background removal from images and videos using U2-Net), and (2) a Django web frontend at backgroundremoverai.com. The pip package is published to PyPI; the Django web app runs on a separate server.

- **Domain**: backgroundremoverai.com
- **Server**: 147.182.206.90
- **GitHub**: github.com/nadermx/backgroundremover

## Repository Structure

```
backgroundremover.app/
├── backgroundremover/          # Pip package source
│   ├── cmd/cli.py              # CLI entry point (backgroundremover command)
│   ├── cmd/server.py           # Flask-based API server (backgroundremover-server)
│   ├── bg.py                   # Core removal logic (U2-Net inference)
│   ├── utilities.py            # Video processing, alpha matting helpers
│   └── u2net/                  # U2-Net model architecture and detection
├── models/                     # Pre-trained U2-Net model weights (.pth)
├── accounts/                   # Django accounts app (pyc only in repo, source on server)
├── finances/                   # Django payments app (pyc only in repo, source on server)
├── translations/               # i18n management commands (seed_translations, run_translation)
├── contact_messages/           # Django contact form (pyc only in repo, source on server)
├── app/                        # Django app config and templatetags
├── ansible/                    # Server management
│   ├── servers                 # Inventory (147.182.206.90)
│   └── group_vars/all          # Credentials (gitignored)
├── config.py                   # Django settings (gitignored, different local vs production)
├── setup.py                    # Pip package setup (version 0.4.1)
└── requirements.txt            # Pip package deps (torch, torchvision, moviepy, Pillow, etc.)
```

**Note**: The Django web app source files (`accounts/`, `finances/`, `contact_messages/` .py files) are NOT tracked in git -- only `.pyc` and migration files exist locally. The full Django source lives on the server at `/home/www/backgroundremover`.

## Architecture

### Pip Package
- **CLI**: `backgroundremover -i input.jpg -o output.png` (image), `backgroundremover -i input.mp4 -tv -o output.mov` (video)
- **Models**: U2-Net, U2-Net Human Seg, U2-NetP (lightweight)
- **Features**: Alpha matting, transparent video (ProRes 4444), GPU acceleration, multi-worker video processing
- **Flask server**: `backgroundremover-server` for HTTP API

### Django Web App
- **Stack**: Django + PostgreSQL + Redis + Gunicorn/Nginx/Supervisor
- **Auth**: `accounts.CustomUser` (email-based, credits system)
- **Payments**: Stripe subscriptions (4 plans: $3/300cr, $6/600cr, $54/5400cr, $184/18400cr)
- **Models**: `FileUrlVal` (tracks processed files per user/IP), `Dicty`/`KeyVal` (job tracking)
- **Processing**: Sends jobs to GPU API (`api.backgroundremoverai.com` -> GPU server 38.248.6.142)
- **Admin**: `/admin/`
- **Email**: Mailgun (not self-hosted Postfix)
- **i18n**: Translation system via translateapi batch API
- **GPU Credit Security**: Uses `GPU_SHARED_SECRET` for credit callback validation
- **Special**: `www.` prefix required for credit callbacks (`https://www.backgroundremoverai.com`)

### File Limits
- Free images: 25MB
- Free video: 100MB
- Pro users: 2GB

## Development

```bash
cd /home/john/backgroundremover.app
source venv/bin/activate

# Run Django web app locally
python manage.py runserver 8050

# Test CLI
backgroundremover -i input.jpg -o output.png
backgroundremover -i input.mp4 -tv -o output.mov

# Translation commands
python manage.py seed_translations
python manage.py run_translation
```

## Deployment

No `gitpull.yml` exists. Deploy via direct SSH:

```bash
sshpass -p 'N@%Yu44V' ssh -o StrictHostKeyChecking=no backgroundremover@147.182.206.90 "cd /home/www/backgroundremover && git pull && echo 'N@%Yu44V' | sudo -S supervisorctl restart backgroundremover"
```

### Publishing pip package
```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

## Server Details

| Field | Value |
|-------|-------|
| IP | 147.182.206.90 |
| SSH User | backgroundremover |
| SSH Password | N@%Yu44V |
| Server Path | /home/www/backgroundremover |
| Domain | backgroundremoverai.com |
| API Domain | api.backgroundremoverai.com (CNAME to GPU server) |
| Database | PostgreSQL |
| Process Manager | Supervisor |

## Cleanup

- 4-hour video file cleanup cron (small 25G disk)
- `purge_old_records` at 3am daily (deletes `FileUrlVal` >30 days)
- Expired page CTA in `{% elif not data %}` template block
