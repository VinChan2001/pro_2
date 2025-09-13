"""
Interactive Sentiment Analysis Demo
Test the trained model on custom movie reviews and explore predictions
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from main import SentimentAnalyzer


class SentimentDemo:
    def __init__(self, model_path='sentiment_model.h5', vocab_size=10000, max_length=250):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Load trained model
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please train the model first by running main.py")
            return
        
        # Load IMDB word index for text processing
        self.word_index = tf.keras.datasets.imdb.get_word_index()
        self.reverse_word_index = {value: key for key, value in self.word_index.items()}
        
        # Create tokenizer with same vocabulary
        self.setup_tokenizer()
    
    def setup_tokenizer(self):
        """Set up tokenizer with IMDB vocabulary"""
        # Load some IMDB data to fit tokenizer
        (x_train, _), _ = tf.keras.datasets.imdb.load_data(num_words=self.vocab_size)
        
        # Convert back to text for tokenizer
        texts = []
        for sequence in x_train[:1000]:  # Use subset for efficiency
            text = ' '.join([self.reverse_word_index.get(i - 3, '?') for i in sequence])
            texts.append(text)
        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
    
    def preprocess_text(self, text):
        """Clean and preprocess custom text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def predict_sentiment(self, text):
        """Predict sentiment of custom text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        
        # Pad sequence
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length)
        
        # Make prediction
        prediction_prob = self.model.predict(padded_sequence, verbose=0)[0][0]
        sentiment = "Positive" if prediction_prob > 0.5 else "Negative"
        confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
        
        return sentiment, confidence, prediction_prob
    
    def analyze_custom_reviews(self, reviews):
        """Analyze multiple custom reviews"""
        results = []
        
        print(f"🔍 Analyzing {len(reviews)} custom reviews...")
        print("-" * 60)
        
        for i, review in enumerate(reviews):
            sentiment, confidence, prob = self.predict_sentiment(review)
            results.append({
                'review': review,
                'sentiment': sentiment,
                'confidence': confidence,
                'probability': prob
            })
            
            # Display result
            color_code = "💚" if sentiment == "Positive" else "❤️"
            print(f"{color_code} Review {i+1}: {sentiment} ({confidence:.2%} confident)")
            print(f"   Text: {review[:100]}{'...' if len(review) > 100 else ''}")
            print()
        
        return results
    
    def create_word_clouds(self, reviews, predictions):
        """Create word clouds for positive and negative reviews"""
        positive_text = ""
        negative_text = ""
        
        for review, pred in zip(reviews, predictions):
            if pred['sentiment'] == 'Positive':
                positive_text += " " + review
            else:
                negative_text += " " + review
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        if positive_text.strip():
            wordcloud_pos = WordCloud(width=400, height=300, background_color='white',
                                     colormap='Greens').generate(positive_text)
            ax1.imshow(wordcloud_pos, interpolation='bilinear')
            ax1.set_title('Positive Reviews - Word Cloud', fontsize=14, fontweight='bold')
            ax1.axis('off')
        
        if negative_text.strip():
            wordcloud_neg = WordCloud(width=400, height=300, background_color='white',
                                     colormap='Reds').generate(negative_text)
            ax2.imshow(wordcloud_neg, interpolation='bilinear')
            ax2.set_title('Negative Reviews - Word Cloud', fontsize=14, fontweight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('custom_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions_distribution(self, results):
        """Plot distribution of prediction probabilities"""
        probabilities = [r['probability'] for r in results]
        sentiments = [r['sentiment'] for r in results]
        
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confidence vs Sentiment
        plt.subplot(1, 2, 2)
        pos_conf = [r['confidence'] for r in results if r['sentiment'] == 'Positive']
        neg_conf = [r['confidence'] for r in results if r['sentiment'] == 'Negative']
        
        plt.boxplot([pos_conf, neg_conf], labels=['Positive', 'Negative'])
        plt.ylabel('Confidence')
        plt.title('Confidence by Predicted Sentiment')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_sample_reviews(self):
        """Test on a variety of sample movie reviews"""
        sample_reviews = [
            "This movie was absolutely fantastic! Great acting, amazing plot, and beautiful cinematography. I loved every minute of it.",
            "Terrible movie. Waste of time and money. Poor acting and boring storyline. Would not recommend to anyone.",
            "An okay film. Nothing spectacular but decent entertainment for a weekend. Could have been better.",
            "Outstanding performance by the lead actor! This film will definitely win awards. Masterpiece of modern cinema.",
            "I fell asleep halfway through. Very slow paced and confusing plot. Not worth watching.",
            "Brilliant direction and script. This movie made me laugh and cry. Truly emotional journey.",
            "Worst movie I've ever seen. Terrible special effects and laughable dialogue. Complete disaster.",
            "Pretty good movie overall. Solid 7/10. Good for a date night or casual viewing.",
            "This film changed my perspective on life. Incredible storytelling and deep meaningful themes.",
            "Boring and predictable. The ending was obvious from the first scene. Very disappointing."
        ]
        
        print("🎬 Testing on sample movie reviews...")
        results = self.analyze_custom_reviews(sample_reviews)
        
        # Create visualizations
        self.create_word_clouds(sample_reviews, results)
        self.plot_predictions_distribution(results)
        
        return results


def interactive_mode():
    """Interactive mode for testing custom reviews"""
    demo = SentimentDemo()
    
    print("\n🎭 Interactive Sentiment Analysis Mode")
    print("Enter movie reviews to analyze their sentiment!")
    print("Type 'quit' to exit, 'samples' to test sample reviews")
    print("-" * 50)
    
    reviews_collected = []
    
    while True:
        user_input = input("\nEnter a movie review: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'samples':
            demo.test_sample_reviews()
            continue
        elif user_input.lower() == 'analyze' and reviews_collected:
            print(f"\n📊 Analyzing {len(reviews_collected)} collected reviews...")
            results = demo.analyze_custom_reviews(reviews_collected)
            demo.create_word_clouds(reviews_collected, results)
            demo.plot_predictions_distribution(results)
            reviews_collected = []
            continue
        elif not user_input:
            continue
        
        # Analyze single review
        sentiment, confidence, prob = demo.predict_sentiment(user_input)
        reviews_collected.append(user_input)
        
        # Display result with formatting
        if sentiment == "Positive":
            print(f"✅ {sentiment} sentiment ({confidence:.1%} confident)")
            print(f"   Probability score: {prob:.3f}")
        else:
            print(f"❌ {sentiment} sentiment ({confidence:.1%} confident)")
            print(f"   Probability score: {prob:.3f}")
        
        print(f"📝 Reviews collected: {len(reviews_collected)} (type 'analyze' to process all)")


def main():
    print("🎬 Sentiment Analysis Demo")
    print("=" * 40)
    
    while True:
        print("\nChoose an option:")
        print("1. Interactive mode - Enter custom reviews")
        print("2. Test sample reviews")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            interactive_mode()
        elif choice == '2':
            demo = SentimentDemo()
            demo.test_sample_reviews()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()