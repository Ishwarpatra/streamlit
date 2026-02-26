# 🚜 Offroad Terrain Segmentation

AI-powered terrain segmentation for offroad environments using DINOv2 and a custom decoder.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 Features

- **10 Terrain Classes**: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, and Sky
- **State-of-the-art Backbone**: Meta AI's DINOv2 ViT-B/14 with registers
- **Real-time Processing**: Fast inference on CPU
- **Interactive UI**: Built with Streamlit for easy deployment and use
- **Detailed Analytics**: View pixel-level terrain distribution statistics

## 🏗️ Architecture

```
Input Image (3×378×672)
        ↓
DINOv2 ViT-B/14 + Registers (Frozen Backbone)
        ↓
Patch Tokens (27×48×768)
        ↓
Custom ConvNeXt Decoder (Trained)
  ├─ Conv2d(768→256) + BN + GELU
  ├─ Conv2d(256→256) + BN + GELU  
  ├─ Conv2d(256→128) + BN + GELU
  └─ Conv2d(128→10)
        ↓
Segmentation Map (10×378×672)
```

## 📦 Repository Structure

```
offroad-segmentation/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── segmentation_head_best.pth     # Trained model weights (you must add this)
├── train_first.py                 # Training script (optional)
└── README.md                      # This file
```

## 🚀 Quick Start

### Option 1: Streamlit Cloud (Recommended)

1. **Fork this repository** to your GitHub account

2. **Add your trained model checkpoint**:
   - Place your trained model file as `segmentation_head_best.pth` in the repository root
   - If the file is >100MB, use [Git LFS](https://git-lfs.github.com/):
     ```bash
     git lfs install
     git lfs track "*.pth"
     git add .gitattributes
     git add segmentation_head_best.pth
     git commit -m "Add model checkpoint"
     git push
     ```

3. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"!

### Option 2: Local Deployment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/offroad-segmentation.git
   cd offroad-segmentation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model checkpoint**:
   ```bash
   # Place your trained model as segmentation_head_best.pth
   cp /path/to/your/checkpoint.pth segmentation_head_best.pth
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**:
   - The app will automatically open at `http://localhost:8501`

## 🎓 Training Your Own Model

If you want to train your own segmentation model:

1. **Prepare your dataset**:
   ```
   dataset/
   ├── train/
   │   ├── Color_Images/
   │   └── Segmentation/
   └── val/
       ├── Color_Images/
       └── Segmentation/
   ```

2. **Update the training script** (`train_first.py`):
   ```python
   trainset = MaskDataset("path_to_train", transform, mask_transform, True)
   valset = MaskDataset("path_to_val", transform, mask_transform, False)
   ```

3. **Train the model**:
   ```bash
   python train_first.py
   ```

4. **The script will save**:
   - Training checkpoints
   - Best model based on IoU
   - Training curves and statistics

## 📊 Model Performance

The model is optimized for:
- **Mean IoU**: Focus on overall segmentation quality
- **Rare Class Performance**: Special attention to underrepresented classes (Logs, Rocks, Ground Clutter)
- **Balanced Metrics**: Equal importance across all terrain types

## 🔧 Configuration

### Model Parameters
- **Image Size**: 672×378 pixels
- **Patch Size**: 14×14
- **Embedding Dimension**: 768 (ViT-B)
- **Number of Classes**: 10
- **Backbone**: Frozen (not trainable)

### Streamlit Settings
Edit `.streamlit/config.toml` to customize:
- Theme colors
- Upload size limits (default: 10MB)
- Browser settings

## 🐛 Troubleshooting

### Error: "No checkpoint found"
**Solution**: Make sure `segmentation_head_best.pth` exists in your repository root. If deploying to Streamlit Cloud and the file is >100MB, use Git LFS.

### Error: "Out of memory"
**Solution**: Streamlit Cloud has limited resources. Try:
- Reducing batch size in code
- Optimizing model architecture
- Using model quantization

### Slow Loading
**Solution**: The DINOv2 model is downloaded on first run. Subsequent runs will be faster due to caching.

### Upload Size Limit
**Solution**: Increase `maxUploadSize` in `.streamlit/config.toml` (currently set to 10MB).

## 📝 Dependencies

### Python Packages
- `streamlit>=1.28.0` - Web application framework
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision models and transforms
- `Pillow>=10.0.0` - Image processing
- `opencv-python-headless>=4.8.0` - Computer vision utilities
- `numpy>=1.24.0` - Numerical computing

### System Packages
- `libgl1-mesa-glx` - OpenGL support
- `libglib2.0-0` - GLib library

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **DINOv2**: Meta AI Research for the powerful vision backbone
- **Streamlit**: For the amazing web app framework
- **PyTorch**: For the deep learning framework

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for offroad terrain analysis**
