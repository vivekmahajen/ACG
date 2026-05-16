#!/usr/bin/env python3
"""One-time re-authentication script.

Run this to generate a fresh token.json that includes the
youtube.force-ssl scope (required for posting comments).

Usage:
    python reauth.py
"""
import json
import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

TOKEN_FILE = "token.json"
CLIENT_SECRET_FILE = "client_secret.json"

if Path(TOKEN_FILE).exists():
    os.remove(TOKEN_FILE)
    print(f"Deleted old {TOKEN_FILE}")

if not Path(CLIENT_SECRET_FILE).exists():
    print(f"ERROR: {CLIENT_SECRET_FILE} not found. Make sure it is in the ACG folder.")
    raise SystemExit(1)

print("Opening browser for Google sign-in...")
print("Grant access to BOTH scopes when prompted.\n")

flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
creds = flow.run_local_server(port=0)

with open(TOKEN_FILE, "w") as f:
    f.write(creds.to_json())

print(f"\nDone! New {TOKEN_FILE} saved with force-ssl scope.")
print("\nNow encode it and update your GitHub secret:")
print(r'  [Convert]::ToBase64String([IO.File]::ReadAllBytes("token.json")) | Set-Clipboard')
print("\nThen go to GitHub -> Settings -> Secrets -> update TOKEN_JSON_B64")
