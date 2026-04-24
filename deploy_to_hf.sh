#!/bin/bash
# Deploy to Hugging Face Spaces

echo "=========================================="
echo "Deploy to Hugging Face Spaces"
echo "=========================================="
echo ""

# Get user input
read -p "Enter your Hugging Face username: " HF_USERNAME
read -p "Enter your Space name: " SPACE_NAME

echo ""
echo "Adding Hugging Face remote..."

# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME

echo "Remote added!"
echo ""
echo "Now pushing to Hugging Face..."
echo "You will be prompted for:"
echo "  Username: $HF_USERNAME"
echo "  Password: Your HF token (starts with hf_...)"
echo ""

# Push to Hugging Face
git push hf main --force

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "Check your Space at:"
echo "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo "=========================================="
