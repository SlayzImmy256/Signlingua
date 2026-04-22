---
title: Sign Language Translator
emoji: 🤟
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🤟 Sign Language Translator

Break communication barriers with AI-powered sign language recognition and translation!

## 🌟 Features

- **Real-time Sign Recognition** - Upload ASL videos and get instant predictions
- **Multi-language Translation** - Translate to 100+ languages
- **Text-to-Speech** - Hear translations with natural voice
- **14 Sign Classes** - Recognizes 14 different ASL signs
- **Beautiful UI** - Modern, user-friendly interface

## 🚀 How to Use

1. **Upload Video** - Record or upload your sign language video
2. **Select Language** - Choose your desired output language
3. **Translate** - Click the button and get instant results!

## 🛠️ Tech Stack

- **PyTorch** - Deep learning framework
- **MediaPipe** - Hand landmark extraction
- **Gradio** - Web interface
- **Deep Translator** - Multi-language support
- **gTTS** - Text-to-speech

## 📊 Model Details

- **Architecture:** Multi-Layer Perceptron (MLP)
- **Input:** MediaPipe landmarks (258 features)
- **Output:** 14 sign classes
- **Training:** Trained on ASL dataset

## ⚠️ Limitations

- Currently recognizes **single signs only** (not full sentences)
- Best results with clear lighting and hand visibility
- Requires good video quality

## 🎯 Use Cases

- Learning sign language alphabet
- Basic communication assistance
- Educational demonstrations
- Accessibility tool

## 📝 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Built with ❤️ for the deaf and hard-of-hearing community.

---

**Note:** This is a proof-of-concept for educational purposes. For production use, consider training on larger datasets and implementing sentence-level recognition.
