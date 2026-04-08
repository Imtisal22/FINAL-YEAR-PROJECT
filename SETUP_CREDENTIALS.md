# Setting Up Google Drive API Credentials

## Steps (one-time setup ~5 minutes)

### 1. Create a Google Cloud Project
1. Go to https://console.cloud.google.com/
2. Click **Select a project** → **New Project**
3. Name it e.g. `imtisal-pipeline` → **Create**

### 2. Enable the Google Drive API
1. In your new project, go to **APIs & Services → Library**
2. Search for **Google Drive API** → Click it → **Enable**

### 3. Create OAuth2 Credentials
1. Go to **APIs & Services → Credentials**
2. Click **+ Create Credentials → OAuth client ID**
3. If prompted to configure consent screen:
   - User Type: **External** → Create
   - App name: `IMTISAL Pipeline`
   - Add your email as both support email and developer email
   - **Save and Continue** through all steps
4. Back at Create OAuth client ID:
   - Application type: **Desktop app**
   - Name: `IMTISAL Desktop`
   - Click **Create**
5. Click **Download JSON**
6. **Rename the downloaded file to `credentials.json`**
7. **Place it in the `imtisal_project/` folder** (same folder as pipeline.py)

### 4. First Run (Browser Auth)
```
cd imtisal_project
python pipeline.py
```
- A browser window opens asking you to sign in to Google
- Click **Allow** to grant read-only Drive access
- A `token.json` is saved automatically — future runs skip the browser step

### 5. Share Your Drive Folder
Make sure the Google account you log in with has access to the **IMTISAL** folder.
If someone else owns it, they need to share it with your Google account.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `credentials.json not found` | Follow Step 3 above |
| `Folder 'IMTISAL' not found` | Check spelling in `config.py → GDRIVE_FOLDER_NAME` |
| `Access not configured` | Make sure Drive API is enabled (Step 2) |
| `Token has been expired` | Delete `token.json` and re-run |
