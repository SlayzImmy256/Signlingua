# 🚀 Quick Deploy to Hugging Face (5 Minutes)

## Fastest Way - Web Interface

### Step 1: Create Account & Space (2 min)
1. Go to https://huggingface.co/join
2. Create free account
3. Go to https://huggingface.co/spaces
4. Click **"Create new Space"**
5. Settings:
   - Name: `sign-language-translator`
   - SDK: **Gradio**
   - Hardware: **CPU basic (free)**
6. Click **"Create Space"**

### Step 2: Upload Files (2 min)
1. Click **"Files"** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload these files:

**Rename when uploading:**
- `app.py` → `app.py` ✅
- `requirements_hf.txt` → **`requirements.txt`** ⚠️ (rename!)
- `README_HF.md` → **`README.md`** ⚠️ (rename!)
- `.gitattributes` → `.gitattributes` ✅

**Upload folders (drag & drop):**
- `app/` folder
- `src/` folder  
- `models/` folder (with best_model.pt)

### Step 3: Wait for Build (1 min)
1. Hugging Face builds automatically
2. Check **"Logs"** tab
3. Wait for: `Running on public URL`

### Step 4: Test! ✅
1. Click **"App"** tab
2. Your app is live!
3. URL: `https://huggingface.co/spaces/YOUR-USERNAME/sign-language-translator`

---

## ✅ Checklist

- [ ] Hugging Face account created
- [ ] Space created (Gradio SDK)
- [ ] `app.py` uploaded
- [ ] `requirements_hf.txt` uploaded as `requirements.txt`
- [ ] `README_HF.md` uploaded as `README.md`
- [ ] `.gitattributes` uploaded
- [ ] `app/` folder uploaded
- [ ] `src/` folder uploaded
- [ ] `models/best_model.pt` uploaded
- [ ] Build completed (check Logs)
- [ ] App working (test upload)

---

## 🎯 Your App URL

After deployment:
```
https://huggingface.co/spaces/YOUR-USERNAME/sign-language-translator
```

Share it everywhere! 🌟

---

## � Troubleshooting

**Build fails?**
- Check Logs tab for errors
- Verify all files uploaded correctly
- Ensure `requirements.txt` and `README.md` are renamed

**App not loading?**
- Wait 2-3 minutes for build
- Refresh the page
- Check if model file uploaded (1.19 MB)

**MediaPipe errors?**
- Normal! App runs in demo mode for MediaPipe
- Model predictions still work with uploaded videos

---

**That's it! Your app is now live on Hugging Face! 🎉**
