# 🚀 Deploy to Hugging Face - Simple Method

## Method 1: Manual Upload (No Git Required)

### Step 1: Clean Your Space
1. Go to your HF Space: https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME
2. Click **"Files"** tab
3. Delete any nested folders (like `sign-language-translator/`)
4. You should have an empty Space or just a README

### Step 2: Upload Files One by One

**Upload these files to ROOT (not in any folder):**

1. Click **"Add file"** → **"Upload files"**
2. Select and upload: **`app.py`**
3. Commit

4. Click **"Add file"** → **"Upload files"**
5. Select and upload: **`requirements.txt`**
6. Commit

7. Click **"Add file"** → **"Upload files"**
8. Select and upload: **`README.md`**
9. Commit

10. Click **"Add file"** → **"Upload files"**
11. Select and upload: **`.gitattributes`**
12. Commit

### Step 3: Upload Folders

**Upload the `app` folder:**
1. Click **"Add file"** → **"Upload files"**
2. Drag the entire **`app`** folder (not just the file inside)
3. Commit

**Upload the `src` folder:**
1. Click **"Add file"** → **"Upload files"**
2. Drag the entire **`src`** folder
3. Commit

**Upload the `models` folder:**
1. Click **"Add file"** → **"Upload files"**
2. Drag the entire **`models`** folder (contains best_model.pt)
3. Commit
4. This may take a few minutes (1.19 MB file)

### Step 4: Check Build
1. Go to **"Logs"** tab
2. Wait for build to complete
3. Look for: "Running on public URL"

### Step 5: Test
1. Click **"App"** tab
2. Your Gradio interface should appear!

---

## Method 2: Use Git (Faster)

### Prerequisites:
- Git installed on your computer
- Hugging Face account

### Steps:

1. **Get your HF token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token (Write access)
   - Copy the token

2. **Open terminal in your project folder:**
   ```bash
   cd "C:\Users\USER\Signlanuage translator\sign-language-translator"
   ```

3. **Add HF remote and push:**
   ```bash
   # Add Hugging Face as a remote
   git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME
   
   # Push to Hugging Face
   git push hf main
   ```
   
   When prompted:
   - Username: Your HF username
   - Password: Your HF token (paste the token you copied)

4. **Wait for build:**
   - Go to your Space on HF
   - Check "Logs" tab
   - Wait for "Running on public URL"

---

## Method 3: Download from GitHub and Upload

### Steps:

1. **Download your GitHub repo:**
   - Go to: https://github.com/SlayzImmy256/Signlingua
   - Click green **"Code"** button
   - Click **"Download ZIP"**
   - Extract the ZIP file

2. **Upload to Hugging Face:**
   - Follow Method 1 above
   - But use the files from the extracted ZIP

---

## ✅ Final Structure Should Look Like:

```
Your-Space/
├── app.py                    ← At root!
├── requirements.txt          ← At root!
├── README.md
├── .gitattributes
├── app/
│   └── gradio_app.py
├── src/
│   ├── data/
│   ├── models/
│   └── translation/
└── models/
    └── best_model.pt
```

**NOT like this:**
```
Your-Space/
└── sign-language-translator/    ← Wrong! Extra folder
    ├── app.py
    └── ...
```

---

## 🆘 Still Having Issues?

Share your Hugging Face Space URL and I can help debug!
