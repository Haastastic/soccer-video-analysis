# Coaching site

The coaching pages (coaching_tips.py) for invited Google accounts, on Google Cloud Run.

- **Roles.** An admin sees everything and manages people. A coach sees every player. A parent sees team numbers and only the players an admin ticked for them.
- **Signing in.** Any Google account can sign in, but an account that is not a user yet only sees a "Request access" form. Each request notifies the admins: an in-app count, plus an email when SMTP is set up.
- **First admin.** `ADMIN_EMAIL` becomes the first admin the first time that account signs in, as long as no admin exists yet.
- **Where things are stored.** Pages are rendered per request from data in a private bucket. Users, requests, the school and opponent names, and the logos live in Firestore. No names, logos or photos are in git.

```
pipeline (local) -> site_export.py -> data/site_export/ --+
player_photo.py -> data/site_photos/player_NN.jpg --------+-> publish_site.py -> private bucket -> Cloud Run app
                                                                                   Firestore (users, logos) --^
```

## One-time setup (the owner runs these; about 30 minutes)

Pick a project ID, a region (for example `us-central1`) and a bucket name. The examples below use `PROJECT`, `REGION` and `BUCKET`.

```powershell
gcloud auth login
gcloud projects create PROJECT
gcloud billing projects link PROJECT --billing-account ACCOUNT_ID   # Cloud Run needs billing; this site costs cents
gcloud config set project PROJECT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com `
    firestore.googleapis.com secretmanager.googleapis.com storage.googleapis.com
gcloud firestore databases create --location=REGION

# private bucket: uniform access, public access prevented
gcloud storage buckets create gs://BUCKET --location=REGION --uniform-bucket-level-access --public-access-prevention

# the site's own identity: read the bucket, use Firestore, read its secrets. Nothing else.
gcloud iam service-accounts create coaching-site
$SA = "coaching-site@PROJECT.iam.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding gs://BUCKET --member="serviceAccount:$SA" --role=roles/storage.objectViewer
gcloud projects add-iam-policy-binding PROJECT --member="serviceAccount:$SA" --role=roles/datastore.user
```

**Google sign-in.** In the Cloud console, go to Google Auth Platform and do the following:

1. Set up the consent screen. User type: External. Scopes: `openid`, `email` and `profile` only. Then publish the app ("In production"). With only these basic scopes, Google does not require verification. While the app is still in "Testing", only the test users listed there can sign in.
2. Create an OAuth client of type "Web application". You'll add its redirect URI after the first deploy.

**Secrets.**

```powershell
python -c "import secrets; print(secrets.token_hex(32))" | gcloud secrets create site-secret-key --data-file=-
"CLIENT_SECRET" | gcloud secrets create site-google-secret --data-file=-      # from the OAuth client
# optional, to email admins about access requests: a Gmail app password (Google account > Security > App passwords)
"APP_PASSWORD" | gcloud secrets create site-smtp-password --data-file=-
foreach ($s in "site-secret-key", "site-google-secret", "site-smtp-password") {
  gcloud secrets add-iam-policy-binding $s --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor }
```

## Deploy (and redeploy after code changes)

```powershell
python webapp/deploy.py stage        # webapp/_build: the site's code + coaching_html.py only, never data/
gcloud run deploy coaching --source webapp/_build --region REGION --service-account $SA `
    --allow-unauthenticated --max-instances 2 --memory 1Gi `
    --set-env-vars "ADMIN_EMAIL=YOUR_GOOGLE_EMAIL,SITE_BUCKET=BUCKET,GOOGLE_CLIENT_ID=CLIENT_ID" `
    --set-env-vars "NOTIFY_SMTP_HOST=smtp.gmail.com,NOTIFY_SMTP_USER=YOUR_GMAIL" `
    --set-secrets "SECRET_KEY=site-secret-key:latest,GOOGLE_CLIENT_SECRET=site-google-secret:latest" `
    --set-secrets "NOTIFY_SMTP_PASSWORD=site-smtp-password:latest"
```

- **`--allow-unauthenticated`.** This only lets browsers reach the sign-in page. The app itself checks every page.
- **Redirect URI.** Add `https://<service URL>/auth/callback` to the OAuth client's authorised redirect URIs. The deploy prints the service URL.
- **Without email.** Leave out the two `NOTIFY_SMTP_*` and `site-smtp-password` parts to skip email. Pending requests still show as a count on Admin.

## Publishing data (after each game)

```powershell
gcloud auth application-default login          # once per machine: lets publish_site.py write to the bucket
python site_export.py --runs <every game's windows, as for coaching_tips.py>
python player_photo.py candidates --runs <windows>   # new players, or better photos
python player_photo.py label                         # choose one crop per player (s = no photo, initials instead)
python publish_site.py --bucket BUCKET --dry-run
python publish_site.py --bucket BUCKET
```

The site picks up a new release within a minute. The newest 3 releases are kept.

Then sign in and open Admin to do the following:

- Set the school name and logo, and each game's opponent and logo.
- Add people, or approve access requests. Tick a parent's players when you approve them.

## Local development

```powershell
python publish_site.py --local-dir webapp/_local_bucket --yes
$env:DEV_LOGIN = "1"; $env:ADMIN_EMAIL = "you@example.com"; $env:SITE_LOCAL_DIR = "webapp/_local_bucket"
$env:DB_FILE = "webapp/_local_db.json"
python webapp/app.py                 # http://127.0.0.1:8080, sign in by typing an email (refused on Cloud Run)
pytest -q tests                      # synthetic players only
```

## Security

- **Sessions.** Sessions are signed cookies: `__Host-`, Secure, HttpOnly, SameSite=Lax, 12 hours. The user record is re-read on every request, so switching someone off takes effect at once.
- **Requests.** Every form has a CSRF token. Pages send a CSP with script nonces, `noindex` and `no-store`.
- **Images.** Photos and logos are only served after a permission check, never from public URLs. Uploaded logos are re-encoded as small PNGs.
- **Access requests.** At most one request can be waiting per account, and at most 3 requests per day. The email to admins carries only the requester's address and a link.
- **Audit.** Every admin change and every request is written to the Firestore `audit` collection.
