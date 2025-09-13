# 🎬 Movie Review Sentiment Analysis

A fun NLP machine learning project that analyzes the sentiment of movie reviews using deep learning with TensorFlow and LSTM networks.

## 🌟 Features

- **LSTM Neural Network**: Deep learning model with embedding and LSTM layers
- **Text Preprocessing**: Advanced text cleaning and tokenization
- **Interactive Training**: Progress tracking with early stopping and learning rate reduction
- **Visualizations**: Training history plots, word clouds, and confusion matrices
- **Interactive Demo**: Test the trained model on custom movie reviews
- **Easy to Use**: Simple setup and clear documentation

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   ```bash
   python main.py
   ```
   This will:
   - Download the IMDB movie reviews dataset automatically
   - Train the LSTM neural network for 10 epochs
   - Generate visualization plots and word clouds
   - Save the trained model as `sentiment_model.h5`

3. **Test the Model**
   ```bash
   python demo.py
   ```
   Interactive demo with options to:
   - Test on custom movie reviews
   - Analyze sentiment predictions with confidence scores
   - Visualize word importance and generate word clouds

## 📊 Model Architecture

```
Input Text → Tokenization → Padding
    ↓
Embedding Layer (vocab_size → 128)
    ↓
Dropout(0.2)
    ↓
LSTM(64) → Dropout(0.5) → Return Sequences
    ↓
LSTM(32) → Dropout(0.5)
    ↓
Dense(32, ReLU) → Dropout(0.5)
    ↓
Dense(1, Sigmoid) → Binary Classification
```

**Total Parameters**: ~1.2M

## 🎯 Performance

- **Training Accuracy**: ~90-95%
- **Validation Accuracy**: ~87-90%
- **Training Time**: ~10-15 minutes on CPU
- **Dataset**: IMDB 50k movie reviews (25k train, 25k test)

## 📁 Project Structure

```
pro_2/
├── main.py           # Main training script
├── demo.py           # Interactive demo
├── requirements.txt  # Python dependencies
├── README.md        # This file
├── sentiment_model.h5 # Trained model (after training)
├── training_history.png # Training plots
├── word_clouds.png  # Word cloud visualizations
├── confusion_matrix.png # Model evaluation
└── *.png            # Generated visualization files
```

## 🔧 Customization

### Training Parameters
Edit `main.py` to modify:
- **Epochs**: Change `epochs=10` in the `train_model()` function
- **Batch Size**: Modify `batch_size=32` in training
- **Vocabulary Size**: Adjust `vocab_size=10000` in SentimentAnalyzer
- **Sequence Length**: Change `max_length=250` for input padding
- **Model Architecture**: Modify the `build_model()` method

### Network Architecture
The model uses:
- **Embedding**: Word embeddings for text representation
- **LSTM**: Long Short-Term Memory for sequence processing
- **Regularization**: Dropout layers and early stopping
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy

## 📈 Generated Files

After training, you'll get:
- `sentiment_model.h5` - Trained LSTM model
- `training_history.png` - Loss and accuracy curves
- `word_clouds.png` - Word clouds for positive/negative reviews
- `confusion_matrix.png` - Model evaluation metrics
- `sample_predictions.png` - Sample predictions with confidence

## 🎮 Demo Features

The interactive demo (`demo.py`) offers:

1. **Custom Review Analysis**: Enter your own movie reviews for sentiment analysis
2. **Confidence Scoring**: Get probability scores for positive/negative sentiment
3. **Batch Testing**: Analyze multiple reviews at once
4. **Visualization**: Generate word clouds and see prediction explanations

## 💡 Tips for Best Results

1. **Review Quality**: For best results with custom reviews:
   - Use complete sentences with proper grammar
   - Include specific opinions about the movie
   - Avoid very short or ambiguous text
   - Use natural language (not formal or technical)

2. **Model Performance**: The model performs best on:
   - English movie reviews
   - Reviews similar to IMDB style
   - Text with clear sentiment indicators

## 🧠 Learning Outcomes

This project demonstrates:
- **Natural Language Processing**: Text preprocessing, tokenization
- **Deep Learning with TensorFlow**: LSTM networks, embeddings
- **Sentiment Analysis**: Binary classification, feature extraction
- **Model Evaluation**: Confusion matrices, classification reports
- **Text Visualization**: Word clouds, prediction analysis
- **Interactive Development**: User-friendly demos

## 🔮 Potential Improvements

- Add attention mechanisms for better performance
- Implement BERT or transformer models
- Add support for multi-class sentiment (very positive, positive, neutral, negative, very negative)
- Create a web interface with Flask/Django
- Add model ensemble techniques
- Implement transfer learning with pre-trained embeddings
- Add support for other languages

---

**Have fun exploring machine learning! 🎉**