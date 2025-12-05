# HairMixer Deployment Guide - DigitalOcean App Platform

This guide walks you through deploying HairMixer to DigitalOcean App Platform.

## Prerequisites

1. DigitalOcean account with student credits ($200)
2. GitHub account with your repository pushed
3. Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

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
   Save this key - you'll need it in Step 3.

## Step 2: Create App on DigitalOcean

### Using the Dashboard (Recommended)

1. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
2. Click **Create App**
3. Connect your GitHub account and select `HairMixer` repository
4. Select `main` branch
5. DigitalOcean will auto-detect the Django app

## Step 3: Configure Environment Variables

In the App Platform dashboard, add these environment variables:

| Variable               | Value                  | Type       |
| ---------------------- | ---------------------- | ---------- |
| `DJANGO_SECRET_KEY`    | (your generated key)   | **Secret** |
| `DJANGO_DEBUG`         | `false`                | Plain      |
| `DJANGO_ALLOWED_HOSTS` | `*.ondigitalocean.app` | Plain      |
| `CORS_ALLOW_ALL`       | `false`                | Plain      |
| `CORS_ALLOWED_ORIGINS` | Your frontend URL      | Plain      |
| `CSRF_TRUSTED_ORIGINS` | Your frontend URL      | Plain      |
| `GEMINI_API_KEY`       | (your Gemini API key)  | **Secret** |
| `GEMINI_MODEL_NAME`    | `gemini-2.0-flash`     | Plain      |

## Step 4: Configure Build Settings

### Backend Component

- **Source Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
- **Run Command**: `gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 backend.wsgi:application`
- **HTTP Port**: `8000`

### Frontend Component

- **Source Directory**: `frontend`
- **Build Command**: `npm install && npm run build`
- **Output Directory**: `build`
- **Environment Variable**: `REACT_APP_API_URL=https://YOUR-BACKEND-URL/api`

## Step 5: Deploy

1. Click **Deploy** in the DigitalOcean dashboard
2. Wait for build to complete (10-15 minutes for first deploy)
3. Check the deployment logs for any errors

## Step 6: Post-Deployment

### Create Admin User

Connect to the console in DigitalOcean and run:

```bash
python manage.py createsuperuser
```

## Troubleshooting

### Build Fails

- Check that all dependencies are in `requirements.txt`
- Verify Python version compatibility

### CORS Errors

- Verify `CORS_ALLOWED_ORIGINS` includes your frontend URL
- Make sure to use `https://` (not `http://`)

### ML Model Errors (Out of Memory)

- The basic-xxs instance may not have enough RAM
- Upgrade to basic-xs ($12/month) if needed

## Estimated Costs

| Component | Instance    | Monthly Cost    |
| --------- | ----------- | --------------- |
| Backend   | basic-xxs   | ~$5-7           |
| Frontend  | Static Site | **Free**        |
| **Total** |             | **~$5-7/month** |

With your $200 student credit, you have ~28+ months of deployment!
