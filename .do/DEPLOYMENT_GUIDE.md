# HairMixer Deployment Guide - DigitalOcean App Platform

This guide walks you through deploying HairMixer to DigitalOcean App Platform.

## Prerequisites

1. DigitalOcean account with student credits ($200)
2. GitHub account with your repository pushed
3. Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## Step 1: Prepare Your Repository

1. **Push your code to GitHub**:

   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Generate a Django Secret Key**:
   ```bash
   python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
   ```
   Save this key - you'll need it later.

---

## Step 2: Create App on DigitalOcean

1. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
2. Click **Create App**
3. Connect your GitHub account
4. Select `HairMixer` repository
5. Select your branch (e.g., `main` or `deployment`)
6. Click **Next**

---

## Step 3: Configure Backend Component

DigitalOcean will auto-detect your Python/Django app. Configure it as follows:

### 3.1 Resource Settings

Click on the detected component (`hairmixer`) to edit:

| Setting              | Value                      |
| -------------------- | -------------------------- |
| **Name**             | `hairmixer` (or `backend`) |
| **Resource type**    | Web Service                |
| **Source Directory** | `backend`                  |

### 3.2 Deployment Settings

Click **"Edit"** next to Deployment settings:

| Setting           | Value                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| **Build command** | `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate` |
| **Run command**   | `gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 backend.wsgi:application`                         |

### 3.3 Network Settings

Click **"Edit"** next to Network:

| Setting              | Value                |
| -------------------- | -------------------- |
| **Public HTTP port** | `8000` ⚠️ Important! |

### 3.4 Environment Variables

Click **"Edit"** next to Environment variables and add:

| Variable               | Value                          | Type          |
| ---------------------- | ------------------------------ | ------------- |
| `DJANGO_SECRET_KEY`    | (your generated key)           | **Encrypted** |
| `DJANGO_DEBUG`         | `false`                        | Plain         |
| `DJANGO_ALLOWED_HOSTS` | `*.ondigitalocean.app`         | Plain         |
| `CORS_ALLOW_ALL`       | `false`                        | Plain         |
| `CORS_ALLOWED_ORIGINS` | `https://*.ondigitalocean.app` | Plain         |
| `CSRF_TRUSTED_ORIGINS` | `https://*.ondigitalocean.app` | Plain         |
| `GEMINI_API_KEY`       | (your Gemini API key)          | **Encrypted** |

### 3.5 Instance Size

| Setting        | Recommended              |
| -------------- | ------------------------ |
| **Size**       | Basic ($12/mo - 1GB RAM) |
| **Containers** | 1                        |

> ⚠️ If you get out-of-memory errors with ML models, upgrade to 2GB RAM.

---

## Step 4: Add Frontend Component

Click **"+ Add Resource"** → **"Create Resource from Source Code"**

### 4.1 Resource Settings

| Setting              | Value                   |
| -------------------- | ----------------------- |
| **Name**             | `frontend`              |
| **Resource type**    | **Static Site** (FREE!) |
| **Source Directory** | `frontend`              |

### 4.2 Build Settings

| Setting              | Value                          |
| -------------------- | ------------------------------ |
| **Build command**    | `npm install && npm run build` |
| **Output directory** | `build`                        |

### 4.3 Environment Variables

| Variable            | Value                       |
| ------------------- | --------------------------- |
| `REACT_APP_API_URL` | `https://${APP_DOMAIN}/api` |

> **Note**: Replace `${APP_DOMAIN}` with your actual app URL after first deployment, or use the internal service URL.

---

## Step 5: Review and Deploy

1. Review the **Summary** panel:
   - Backend: ~$12/month (Web Service)
   - Frontend: **FREE** (Static Site)
2. Choose **Datacenter region** (Singapore recommended for Asia)

3. Click **"Create app"**

4. Wait for deployment (10-15 minutes for first deploy)

---

## Step 6: Post-Deployment Setup

### 6.1 Get Your App URL

After deployment, your app URL will be:

```
https://hairmixer-xxxxx.ondigitalocean.app
```

### 6.2 Update Frontend API URL (if needed)

If you used a placeholder, update `REACT_APP_API_URL` with your actual backend URL:

```
https://hairmixer-xxxxx.ondigitalocean.app/api
```

### 6.3 Create Admin User

1. Go to your app in DigitalOcean dashboard
2. Click **Console** tab
3. Run:
   ```bash
   python manage.py createsuperuser
   ```

---

## Troubleshooting

### Build Fails

- Check `requirements.txt` has all dependencies
- Verify `gunicorn` and `whitenoise` are included
- Check deployment logs for specific errors

### "No module named 'backend'" Error

- Ensure **Source Directory** is set to `backend`
- Verify `backend/backend/wsgi.py` exists

### Port Issues / 502 Error

- Ensure HTTP port is `8000` (not `8080`)
- Check run command uses `--bind 0.0.0.0:8000`

### CORS Errors

- Add `https://` prefix to all origins
- Use `*.ondigitalocean.app` wildcard for testing

### Static Files Not Loading

- Verify `collectstatic` is in build command
- Check WhiteNoise is in MIDDLEWARE

### ML Model Out of Memory

- Upgrade to 2GB RAM instance ($24/month)
- Or reduce workers: `--workers 1`

---

## Estimated Costs

| Component | Type              | Monthly Cost  |
| --------- | ----------------- | ------------- |
| Backend   | Web Service (1GB) | $12           |
| Frontend  | Static Site       | **FREE**      |
| **Total** |                   | **$12/month** |

With $200 student credits → **16+ months of deployment!**

---

## Updating Your App

Push changes to GitHub and DigitalOcean auto-deploys:

```bash
git add .
git commit -m "Update feature"
git push origin main
```
