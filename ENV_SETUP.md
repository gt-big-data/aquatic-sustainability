# Setting Environment Variables in Google Cloud Platform

There are two ways to set environment variables for your Cloud Run service:

## Method 1: Set via Cloud Build (Recommended for CI/CD)

You can pass environment variables as substitution variables when triggering the build:

```bash
gcloud builds submit --config cloudbuild.yaml \
    --substitutions=_SECRET_KEY=your-secret-key,_GOOGLE_MAPS_API_KEY=your-api-key,_SUPABASE_URL=your-url,_SUPABASE_ANON_KEY=your-key,_MONGODB_URI=your-uri
```

**Note:** For sensitive values, use Secret Manager instead (see Method 3 below).

## Method 2: Set via Google Cloud Console (Web UI)

### Step-by-Step Instructions:

1. **Navigate to Cloud Run:**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Select your project
   - Navigate to **Cloud Run** in the left sidebar (under "Serverless")

2. **Select Your Service:**
   - Click on the service name: `aquatic-sustainability`

3. **Edit the Service:**
   - Click the **"EDIT & DEPLOY NEW REVISION"** button at the top

4. **Add Environment Variables:**
   - Scroll down to the **"Variables & Secrets"** section
   - Click **"ADD VARIABLE"** for each environment variable
   - Add the following variables:
     - `SECRET_KEY` = your secret key
     - `GOOGLE_MAPS_API_KEY` = your Google Maps API key
     - `SUPABASE_URL` = your Supabase project URL
     - `SUPABASE_ANON_KEY` = your Supabase anonymous key
     - `MONGODB_URI` = your MongoDB connection string (if using)

5. **Deploy:**
   - Scroll to the bottom and click **"DEPLOY"**

### Visual Guide:
```
Cloud Run → aquatic-sustainability → EDIT & DEPLOY NEW REVISION → 
Variables & Secrets → ADD VARIABLE → [Enter name and value] → DEPLOY
```

## Method 3: Set via gcloud CLI (Command Line)

After your service is deployed, you can update environment variables using the gcloud CLI:

```bash
gcloud run services update aquatic-sustainability \
    --region=us-central1 \
    --set-env-vars="SECRET_KEY=your-secret-key,GOOGLE_MAPS_API_KEY=your-api-key,SUPABASE_URL=your-url,SUPABASE_ANON_KEY=your-key,MONGODB_URI=your-uri"
```

Or set them individually:

```bash
gcloud run services update aquatic-sustainability \
    --region=us-central1 \
    --update-env-vars="SECRET_KEY=your-secret-key"
```

## Method 4: Using Secret Manager (Best for Sensitive Data)

For production, use Google Secret Manager for sensitive values:

1. **Create Secrets:**
```bash
echo -n "your-secret-key" | gcloud secrets create secret-key --data-file=-
echo -n "your-supabase-key" | gcloud secrets create supabase-anon-key --data-file=-
```

2. **Grant Cloud Run Access:**
```bash
gcloud secrets add-iam-policy-binding secret-key \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

3. **Update Cloud Build YAML:**
   - Modify `cloudbuild.yaml` to use `--set-secrets` instead of `--set-env-vars`
   - Example: `--set-secrets="SECRET_KEY=secret-key:latest"`

## Verify Environment Variables

To check if environment variables are set correctly:

```bash
gcloud run services describe aquatic-sustainability \
    --region=us-central1 \
    --format="value(spec.template.spec.containers[0].env)"
```

Or view them in the Cloud Console:
- Cloud Run → aquatic-sustainability → REVISIONS tab → Click on a revision → View "Environment variables" section

## Important Notes

- Environment variables are set per revision
- When you deploy a new revision, it will use the environment variables from the previous revision unless you change them
- Changes to environment variables create a new revision
- The service will restart when environment variables are updated
- For sensitive data, always use Secret Manager instead of plain environment variables


