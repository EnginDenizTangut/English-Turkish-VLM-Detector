# English-Turkish VLM Detector

A sophisticated Vision Language Model (VLM) detector that combines natural language processing with computer vision to detect objects in images based on Turkish language queries. This project uses Ollama's Llama 3.1 model to understand Turkish natural language commands and YOLOv8 to perform precise object detection.

## 🌟 Features

- **🤖 Advanced LLM Integration**: Uses Ollama's Llama 3.1:latest model for natural language understanding
- **🎯 YOLOv8 Object Detection**: State-of-the-art object detection with 80 COCO classes
- **🇹🇷 Turkish Language Support**: Native Turkish language query processing
- **📸 Image Processing**: Advanced image processing and visualization capabilities
- **🎨 Visual Output**: Automatic bounding box drawing with confidence scores
- **🔄 Smart Class Mapping**: Intelligent mapping between Turkish queries and English COCO classes
- **🌈 Custom Colors**: Support for custom bounding box colors based on Turkish color names
- **🎯 Color Filtering**: Intelligent color-based object filtering (e.g., "beyaz kedileri göster" - show only white cats)

## 🚀 How It Works

1. **Input Processing**: Takes a Turkish natural language query (e.g., "mavi arabaları göster" - show blue cars)
2. **Color Detection**: Extracts color preferences from the query
3. **LLM Analysis**: Uses Llama 3.1 to map Turkish terms to COCO class names
4. **Object Detection**: YOLOv8 detects all objects in the image
5. **Class Filtering**: Filters detections based on the LLM's class mapping
6. **Color Filtering**: If color specified, filters objects by actual color analysis
7. **Visualization**: Draws bounding boxes around matching objects with custom colors and confidence scores

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama installed on your system
- Sufficient RAM for running Llama 3.1 model

## 🛠️ Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install and Setup Ollama

1. **Install Ollama**:

   - Visit [https://ollama.ai/](https://ollama.ai/)
   - Download and install Ollama for your operating system

2. **Pull the Llama 3.1 Model**:
   ```bash
   ollama pull llama3.1:latest
   ```

### Step 3: Verify Installation

```bash
python main.py
```

## 💻 Usage

### 🖥️ GUI Interface (Recommended)

For the best user experience, use the modern GUI interface:

```bash
python gui.py
```

**GUI Features:**

- **📸 Dual Panel Display**: Original image on the left, detection results on the right
- **📁 Easy File Selection**: Browse button for image selection
- **💬 Smart Prompt Input**: Long text field with example prompts
- **🎨 Real-time Visualization**: Instant display of detection results
- **💾 Save Results**: Save detection results as images
- **🔄 Modern Interface**: Clean, responsive design

#### GUI Screenshot

```
┌─────────────────────────────────────────────────────────────────┐
│                    🎯 English-Turkish VLM Detector             │
├─────────────────────────┬───────────────────────────────────────┤
│  📸 Original Image      │  🎨 Detection Result                 │
│  ┌─────────────────┐   │  ┌─────────────────────────────────┐ │
│  │                 │   │  │                                 │ │
│  │   [Your Image]  │   │  │    [Detection Results]         │ │
│  │                 │   │  │                                 │ │
│  └─────────────────┘   │  └─────────────────────────────────┘ │
├─────────────────────────┴───────────────────────────────────────┤
│ 📁 Image File: [Browse] [car1.webp                    ]        │
│ 💬 Detection Prompt: [mavi arabaları göster           ]        │
│ 💡 Examples: [mavi arabaları göster] [kırmızı kedileri bul]... │
│ 🔍 Detect Objects  🗑️ Clear  💾 Save Result                   │
│ Status: Ready - Select an image and enter a prompt             │
└─────────────────────────────────────────────────────────────────┘
```

### 🖥️ Command Line Interface

For command-line usage:

```bash
python main.py
```

When prompted:

1. Enter the path to your image file
2. Enter your query in Turkish (e.g., "mavi arabaları göster", "kırmızı kedileri bul")

### Example Session

```
Görüntü dosyasının yolunu girin: car1.webp
Ne aramak istiyorsunuz? (örn: 'mavi arabaları göster', 'kırmızı kedileri bul'): mavi arabaları göster
```

## 🎯 Supported Object Categories

The system supports all 80 COCO dataset classes with intelligent Turkish-to-English mapping:

### 🚗 Vehicles

- **Turkish**: "araba", "otomobil", "taşıt", "vasıta" → **English**: car
- **Turkish**: "kamyon", "tır", "yük aracı" → **English**: truck
- **Turkish**: "otobüs", "şehir otobüsü" → **English**: bus
- **Turkish**: "motosiklet", "moto", "motor" → **English**: motorcycle
- **Turkish**: "bisiklet", "velespit", "pedal" → **English**: bicycle

### 🐱 Animals

- **Turkish**: "kedi", "pisi", "miyav" → **English**: cat
- **Turkish**: "köpek", "it", "hav hav" → **English**: dog
- **Turkish**: "kuş", "kanatlı" → **English**: bird
- **Turkish**: "at", "beygir" → **English**: horse
- **Turkish**: "inek", "sığır" → **English**: cow

### 🍎 Food Items

- **Turkish**: "elma", "kırmızı elma" → **English**: apple
- **Turkish**: "muz", "sarı meyve" → **English**: banana
- **Turkish**: "pizza", "italyan yemeği" → **English**: pizza
- **Turkish**: "pasta", "kek", "tatlı" → **English**: cake

### 🪑 Furniture & Objects

- **Turkish**: "sandalye", "oturak", "koltuk" → **English**: chair
- **Turkish**: "masa", "yemek masası" → **English**: dining table
- **Turkish**: "televizyon", "tv", "ekran" → **English**: tv
- **Turkish**: "laptop", "dizüstü" → **English**: laptop

### 👥 People

- **Turkish**: "insan", "kişi", "adam", "kadın", "çocuk" → **English**: person

## 🌈 Supported Colors

The system supports custom bounding box colors with Turkish color names:

### Basic Colors

- **kırmızı** (red) - `(0, 0, 255)`
- **mavi** (blue) - `(255, 0, 0)`
- **yeşil** (green) - `(0, 255, 0)`
- **sarı** (yellow) - `(0, 255, 255)`
- **mor** (purple) - `(255, 0, 255)`
- **turuncu** (orange) - `(0, 165, 255)`
- **pembe** (pink) - `(203, 192, 255)`
- **siyah** (black) - `(0, 0, 0)`
- **beyaz** (white) - `(255, 255, 255)`
- **gri** (gray) - `(128, 128, 128)`

### Advanced Colors

- **kahverengi** (brown) - `(42, 42, 165)`
- **lacivert** (navy blue) - `(139, 0, 0)`
- **altın** (gold) - `(0, 215, 255)`
- **gümüş** (silver) - `(192, 192, 192)`
- **cyan** - `(255, 255, 0)`
- **magenta** - `(255, 0, 255)`

### Color Usage Examples

- "mavi arabaları göster" - Show only blue cars
- "kırmızı kedileri bul" - Find only red cats
- "yeşil sandalyeleri tespit et" - Detect only green chairs
- "sarı meyveleri göster" - Show only yellow fruits
- "beyaz kedileri göster" - Show only white cats
- "arabaları göster" - Show all cars (no color filtering)

## 📁 Project Structure

```
English-Turkish-VLM-Detector/
├── main.py                 # Command-line application
├── gui.py                  # Modern GUI application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── yolov8n.pt            # YOLOv8 model weights (auto-downloaded)
├── car1.webp             # Sample image
├── chairs.jpg            # Sample image
├── traffic.webp          # Sample image
└── output_detection.jpg  # Output image with detections
```

## 🔧 Technical Details

### Core Components

1. **VLMDetector Class**: Main detector class that handles the entire pipeline
2. **Object Detection**: Uses YOLOv8 for fast and accurate object detection
3. **LLM Integration**: Ollama API integration for natural language processing
4. **Class Mapping**: Intelligent mapping system for Turkish-English class translation
5. **Visualization**: OpenCV-based bounding box drawing and labeling

### Dependencies

- `ultralytics==8.0.196` - YOLOv8 implementation
- `ollama==0.1.7` - Ollama API client
- `opencv-python==4.8.1.78` - Computer vision library
- `Pillow==10.0.1` - Image processing
- `numpy==1.24.3` - Numerical computing
- `requests==2.31.0` - HTTP library

## 📊 Output

The system generates:

- **Console Output**: Detection results, confidence scores, and processing information
- **Visual Output**: `output_detection.jpg` with bounding boxes around detected objects
- **Class Information**: Detailed mapping between Turkish queries and detected classes

## 🎨 Example Output

```
Görüntü işleniyor: car1.webp
Kullanıcı sorgusu: mavi arabaları göster
Tespit edilen renk: mavi
LLM sınıf eşleştirmesi: car
Parse edilen sınıflar: ['car']
Tespit edilen nesneler: ['car', 'car']
Sonuç görüntüsü kaydedildi: output_detection.jpg
Bounding box rengi: mavi
```

## 🚀 Advanced Features

### Smart Class Mapping

The system includes an extensive mapping dictionary that handles:

- Synonyms and variations in Turkish
- Contextual understanding
- Multiple word combinations
- Category-based grouping

### Confidence Scoring

Each detection includes confidence scores for reliability assessment.

### Error Handling

Robust error handling for:

- Invalid image paths
- LLM connection issues
- Model loading failures

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Additional Turkish language mappings
- Performance improvements
- New features
- Bug fixes

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Ollama not found**: Ensure Ollama is installed and running
2. **Model not found**: Run `ollama pull llama3.1:latest`
3. **CUDA errors**: Ensure proper GPU drivers are installed
4. **Memory issues**: Close other applications to free up RAM

### Getting Help

If you encounter any issues:

1. Check that all dependencies are installed correctly
2. Verify Ollama is running: `ollama list`
3. Test with a simple image and query
4. Check the console output for error messages

## 🔮 Future Enhancements

- Support for video processing
- Real-time webcam detection
- Additional language support
- Custom model training capabilities
- Web interface development
