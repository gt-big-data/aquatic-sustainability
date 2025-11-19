# Google Cloud Build Deployment Guide

This guide explains how to deploy the Aquatic Sustainability application to Google Cloud using Cloud Build 1st gen.

## Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with billing enabled
2. **gcloud CLI**: Install and authenticate with `gcloud auth login`
3. **Enable APIs**: Enable the following APIs in your project:
   - Cloud Build API
   - Cloud Run API
   - Container Registry API (or Artifact Registry API if using the alternative config)

## Setup Options

### Option 1: Container Registry (GCR) - Default

The `cloudbuild.yaml` file uses Google Container Registry (GCR).

**Setup Steps:**

1. Enable required APIs:
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

2. Submit the build:
```bash
gcloud builds submit --config cloudbuild.yaml
```

### Option 2: Artifact Registry (Recommended)

The `cloudbuild-artifact-registry.yaml` file uses Artifact Registry (newer, recommended).

**Setup Steps:**

1. Enable required APIs:
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

2. Create an Artifact Registry repository:
```bash
gcloud artifacts repositories create aquatic-sustainability-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for Aquatic Sustainability"
```

3. Submit the build with substitution variables:
```bash
gcloud builds submit --config cloudbuild-artifact-registry.yaml \
    --substitutions=_REGION=us-central1,_REPOSITORY=aquatic-sustainability-repo
```

Or rename the file to `cloudbuild.yaml` and set default substitutions:
```bash
gcloud builds submit --substitutions=_REGION=us-central1,_REPOSITORY=aquatic-sustainability-repo
```

## Environment Variables

The application requires the following environment variables. Set them in Cloud Run:

```bash
gcloud run services update aquatic-sustainability \
    --region=us-central1 \
    --set-env-vars="SECRET_KEY=your-secret-key,GOOGLE_MAPS_API_KEY=your-api-key,SUPABASE_URL=your-supabase-url,SUPABASE_ANON_KEY=your-supabase-key,MONGODB_URI=your-mongodb-uri"
```

Or set them during deployment by modifying the `cloudbuild.yaml` deploy step to include `--set-env-vars`.

## Customization

### Change Region

Edit the `--region` parameter in the deploy step. Common regions:
- `us-central1` (Iowa)
- `us-east1` (South Carolina)
- `europe-west1` (Belgium)
- `asia-northeast1` (Tokyo)

### Adjust Resources

Modify the Cloud Run deployment args:
- `--memory`: Memory allocation (e.g., `512Mi`, `1Gi`, `2Gi`)
- `--cpu`: CPU allocation (e.g., `1`, `2`, `4`)
- `--max-instances`: Maximum concurrent instances

### Build Machine Type

Change the `machineType` in the `options` section:
- `E2_HIGHCPU_8`: 8 vCPUs, 8GB RAM (default)
- `E2_HIGHCPU_32`: 32 vCPUs, 32GB RAM (for faster builds)
- `N1_HIGHCPU_8`: Alternative option

## Monitoring

After deployment, monitor your service:
- View logs: `gcloud run services logs read aquatic-sustainability --region=us-central1`
- View service details: `gcloud run services describe aquatic-sustainability --region=us-central1`

## Troubleshooting

1. **Build fails**: Check Cloud Build logs in the Google Cloud Console
2. **Deployment fails**: Ensure Cloud Run API is enabled and you have proper permissions
3. **App crashes**: Check Cloud Run logs and verify environment variables are set correctly
4. **Port issues**: Ensure the app listens on the PORT environment variable (Cloud Run sets this automatically)

## Continuous Deployment

To set up automatic deployments on git push:

1. Connect your repository to Cloud Build:
   - Go to Cloud Build > Triggers in the Google Cloud Console
   - Click "Create Trigger"
   - Connect your repository (GitHub, GitLab, etc.)
   - Set the configuration file to `cloudbuild.yaml`
   - Set branch/tag filters as needed

2. The trigger will automatically build and deploy on each push to the specified branch.

