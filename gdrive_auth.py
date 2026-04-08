"""
Google Drive authentication + file utilities.

First run:  opens a browser window — log in and grant permission.
Subsequent runs: uses the cached token.json silently.
"""

import os
import io
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import CREDENTIALS_FILE, TOKEN_FILE, GDRIVE_FOLDER_NAME, RAW_DIR

# Read-only scope is enough — we only download
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service():
    """Return an authenticated Drive v3 service object."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[auth] Refreshing expired token …")
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"\n[ERROR] '{CREDENTIALS_FILE}' not found.\n"
                    "Steps to fix:\n"
                    "  1. Go to https://console.cloud.google.com/\n"
                    "  2. Create a project → Enable 'Google Drive API'\n"
                    "  3. Credentials → Create → OAuth 2.0 Client ID → Desktop App\n"
                    "  4. Download JSON → rename to 'credentials.json'\n"
                    "  5. Place it next to this script and re-run.\n"
                )
            print("[auth] Opening browser for Google OAuth consent …")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as fh:
            fh.write(creds.to_json())
        print(f"[auth] Token saved → {TOKEN_FILE}")

    service = build("drive", "v3", credentials=creds)
    print("[auth] Google Drive service ready.")
    return service


def find_folder_id(service, folder_name: str) -> str:
    """Return the Drive folder ID for `folder_name` (first match)."""
    query = (
        f"name = '{folder_name}' "
        "and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, parents)",
    ).execute()
    files = results.get("files", [])
    if not files:
        raise FileNotFoundError(
            f"[ERROR] Folder '{folder_name}' not found in your Google Drive.\n"
            "Make sure the folder name matches exactly (case-sensitive)."
        )
    folder_id = files[0]["id"]
    print(f"[drive] Found folder '{folder_name}' → id={folder_id}")
    return folder_id


def list_files_in_folder(service, folder_id: str) -> list[dict]:
    """Return all files (not sub-folders) inside a Drive folder."""
    query = (
        f"'{folder_id}' in parents "
        "and mimeType != 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, mimeType, size, modifiedTime)",
        orderBy="name",
    ).execute()
    files = results.get("files", [])
    print(f"[drive] Found {len(files)} file(s) in folder.")
    for f in files:
        size_kb = int(f.get("size", 0)) // 1024
        print(f"        {f['name']}  ({size_kb} KB)  [{f['mimeType']}]")
    return files


def download_file(service, file_meta: dict) -> str:
    """
    Download a Drive file to RAW_DIR.
    Handles both regular files and Google Sheets (exported as CSV).
    Returns the local file path.
    """
    file_id   = file_meta["id"]
    file_name = file_meta["name"]
    mime_type = file_meta.get("mimeType", "")

    local_path = os.path.join(RAW_DIR, file_name)

    # Google Sheets → export as CSV
    if mime_type == "application/vnd.google-apps.spreadsheet":
        local_path = os.path.join(RAW_DIR, file_name + ".csv")
        request = service.files().export_media(
            fileId=file_id,
            mimeType="text/csv",
        )
    else:
        request = service.files().get_media(fileId=file_id)

    print(f"[download] {file_name} → {local_path} …", end=" ", flush=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    with open(local_path, "wb") as fh:
        fh.write(buf.read())
    print("done")
    return local_path


def download_all(service, folder_name: str = GDRIVE_FOLDER_NAME) -> list[str]:
    """
    Top-level helper: find folder → list files → download all.
    Returns list of local file paths.
    """
    folder_id  = find_folder_id(service, folder_name)
    files_meta = list_files_in_folder(service, folder_id)

    local_paths = []
    for fm in files_meta:
        ext = os.path.splitext(fm["name"])[1].lower()
        mime = fm.get("mimeType", "")
        # Only download CSV, Excel, or Google Sheets
        if ext in (".csv", ".xlsx", ".xls") or "spreadsheet" in mime:
            local_paths.append(download_file(service, fm))
        else:
            print(f"[skip] {fm['name']} (unsupported type: {ext})")

    return local_paths
