"""
Enhanced Interactive Sentiment Analysis Demo
Test the trained model with improved features, logging, and error handling
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import logging
import json
from datetime import datetime
from config import MODEL_CONFIG, PATHS, DEMO_CONFIG, VIZ_CONFIG
import ssl

# Handle SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedSentimentDemo:
    def __init__(self, model_path=None):
        self.model_path = model_path or PATHS['model_file']
        self.vocab_size = MODEL_CONFIG['vocab_size']
        self.max_length = MODEL_CONFIG['max_length']
        
        logger.info(f"Initializing Enhanced Sentiment Demo")
        logger.info(f"Model path: {self.model_path}")
        
        # Load trained model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded successfully from {self.model_path}")
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            print(f"❌ Error loading model: {e}")
            print("Please train the model first by running main.py")
            return
        
        # Load IMDB word index for text processing
        try:
            self.word_index = tf.keras.datasets.imdb.get_word_index()
            self.reverse_word_index = {value: key for key, value in self.word_index.items()}
            logger.info("Word index loaded successfully")
        except Exception as e:
            logger.error(f"Error loading word index: {e}")
            raise
        
        # Create tokenizer with same vocabulary
        self.setup_tokenizer()
        
        # Initialize session data
        self.session_data = {
            'start_time': datetime.now(),
            'predictions_made': 0,
            'results_history': []
        }
    
    def setup_tokenizer(self):
        """Set up tokenizer with IMDB vocabulary"""
        try:
            logger.info("Setting up tokenizer...")
            # Load some IMDB data to fit tokenizer
            (x_train, _), _ = tf.keras.datasets.imdb.load_data(num_words=self.vocab_size)
            
            # Convert back to text for tokenizer
            texts = []
            for sequence in x_train[:1000]:  # Use subset for efficiency
                text = ' '.join([self.reverse_word_index.get(i - 3, '?') for i in sequence])
                texts.append(text)
            
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
            logger.info("Tokenizer setup completed")
        except Exception as e:
            logger.error(f"Error setting up tokenizer: {e}")
            raise
    
    def preprocess_text(self, text):
        """Clean and preprocess custom text with improved handling"""
        if not text or not text.strip():
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            
            # Remove special characters but keep spaces and basic punctuation
            text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            logger.debug(f"Preprocessed text: {text[:100]}...")
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def predict_sentiment(self, text):
        """Predict sentiment of custom text with enhanced error handling"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for prediction")
                return "Unknown", 0.0, 0.5
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                logger.warning("Text became empty after preprocessing")
                return "Unknown", 0.0, 0.5
            
            # Convert to sequence
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            
            if not sequence or not sequence[0]:
                logger.warning("Failed to convert text to sequence")
                return "Unknown", 0.0, 0.5
            
            # Pad sequence
            padded_sequence = pad_sequences(sequence, maxlen=self.max_length)
            
            # Make prediction
            prediction_prob = self.model.predict(padded_sequence, verbose=0)[0][0]
            sentiment = "Positive" if prediction_prob > 0.5 else "Negative"
            confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
            
            # Update session data
            self.session_data['predictions_made'] += 1
            result = {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'probability': prediction_prob,
                'timestamp': datetime.now().isoformat()
            }
            self.session_data['results_history'].append(result)
            
            logger.info(f"Prediction made: {sentiment} ({confidence:.2%})")
            return sentiment, confidence, prediction_prob
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "Error", 0.0, 0.5
    
    def analyze_batch_reviews(self, reviews):
        """Analyze multiple reviews with progress tracking"""
        if not reviews:
            logger.warning("No reviews provided for batch analysis")
            return []
        
        results = []
        
        logger.info(f"Analyzing {len(reviews)} reviews...")
        print(f"🔍 Analyzing {len(reviews)} custom reviews...")
        print("-" * 60)
        
        for i, review in enumerate(reviews):
            try:
                sentiment, confidence, prob = self.predict_sentiment(review)
                results.append({
                    'review': review,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probability': prob
                })
                
                # Display result
                color_code = "💚" if sentiment == "Positive" else "❤️" if sentiment == "Negative" else "❓"
                print(f"{color_code} Review {i+1}: {sentiment} ({confidence:.2%} confident)")
                print(f"   Text: {review[:100]}{'...' if len(review) > 100 else ''}")
                print()
            except Exception as e:
                logger.error(f"Error analyzing review {i+1}: {e}")
                print(f"❌ Error analyzing review {i+1}")
        
        return results
    
    def save_session_data(self):
        """Save session data to JSON file"""
        try:
            session_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert datetime objects to strings for JSON serialization
            session_copy = self.session_data.copy()
            session_copy['start_time'] = session_copy['start_time'].isoformat()
            session_copy['duration_minutes'] = (datetime.now() - self.session_data['start_time']).total_seconds() / 60
            
            with open(session_file, 'w') as f:
                json.dump(session_copy, f, indent=2)
            
            logger.info(f"Session data saved to {session_file}")
            print(f"📊 Session data saved to {session_file}")
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
    
    def create_enhanced_word_clouds(self, reviews, predictions):
        """Create enhanced word clouds with better styling"""
        try:
            positive_text = ""
            negative_text = ""
            
            for review, pred in zip(reviews, predictions):
                if pred['sentiment'] == 'Positive':
                    positive_text += " " + review
                else:
                    negative_text += " " + review
            
            plt.style.use('default')  # Use default style instead of seaborn
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            if positive_text.strip():
                wordcloud_pos = WordCloud(
                    width=600, height=400, 
                    background_color='white',
                    colormap='Greens',
                    max_words=100,
                    relative_scaling=0.5
                ).generate(positive_text)
                ax1.imshow(wordcloud_pos, interpolation='bilinear')
                ax1.set_title('Positive Reviews - Word Cloud', fontsize=16, fontweight='bold', color='green')
                ax1.axis('off')
            
            if negative_text.strip():
                wordcloud_neg = WordCloud(
                    width=600, height=400, 
                    background_color='white',
                    colormap='Reds',
                    max_words=100,
                    relative_scaling=0.5
                ).generate(negative_text)
                ax2.imshow(wordcloud_neg, interpolation='bilinear')
                ax2.set_title('Negative Reviews - Word Cloud', fontsize=16, fontweight='bold', color='red')
                ax2.axis('off')
            
            plt.tight_layout()
            filename = 'enhanced_wordclouds.png'
            plt.savefig(filename, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
            plt.show()
            
            logger.info(f"Enhanced word clouds saved to {filename}")
            print(f"💾 Word clouds saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error creating word clouds: {e}")
            print(f"❌ Error creating word clouds: {e}")
    
    def test_sample_reviews(self):
        """Test on enhanced sample movie reviews"""
        logger.info("Testing sample reviews")
        print("🎬 Testing on sample movie reviews...")
        
        results = self.analyze_batch_reviews(DEMO_CONFIG['sample_reviews'])
        
        # Create visualizations
        self.create_enhanced_word_clouds(DEMO_CONFIG['sample_reviews'], results)
        
        # Print summary
        positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
        negative_count = len(results) - positive_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\n📈 Analysis Summary:")
        print(f"   Positive reviews: {positive_count}/{len(results)}")
        print(f"   Negative reviews: {negative_count}/{len(results)}")
        print(f"   Average confidence: {avg_confidence:.2%}")
        
        return results


def enhanced_interactive_mode():
    """Enhanced interactive mode with better UX"""
    try:
        demo = EnhancedSentimentDemo()
    except Exception as e:
        print(f"❌ Failed to initialize demo: {e}")
        return
    
    print("\n🎭 Enhanced Interactive Sentiment Analysis Mode")
    print("=" * 55)
    print("Commands:")
    print("  • Enter a movie review to analyze")
    print("  • 'samples' - Test on predefined sample reviews")
    print("  • 'history' - Show prediction history")
    print("  • 'save' - Save session data")
    print("  • 'quit' - Exit")
    print("-" * 55)
    
    while True:
        try:
            user_input = input("\n🎬 Enter command or movie review: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                demo.save_session_data()
                print("👋 Thanks for using the sentiment analyzer!")
                break
            elif user_input.lower() == 'samples':
                demo.test_sample_reviews()
                continue
            elif user_input.lower() == 'history':
                if demo.session_data['results_history']:
                    print(f"\n📊 Prediction History ({len(demo.session_data['results_history'])} predictions):")
                    for i, result in enumerate(demo.session_data['results_history'][-5:], 1):  # Show last 5
                        print(f"  {i}. {result['sentiment']} ({result['confidence']:.1%}) - {result['text'][:50]}...")
                else:
                    print("No predictions made yet.")
                continue
            elif user_input.lower() == 'save':
                demo.save_session_data()
                continue
            elif not user_input:
                continue
            
            # Analyze single review
            sentiment, confidence, prob = demo.predict_sentiment(user_input)
            
            # Display result with enhanced formatting
            if sentiment == "Positive":
                print(f"✅ {sentiment} sentiment ({confidence:.1%} confident)")
                print(f"   Probability score: {prob:.3f}")
            elif sentiment == "Negative":
                print(f"❌ {sentiment} sentiment ({confidence:.1%} confident)")
                print(f"   Probability score: {prob:.3f}")
            else:
                print(f"❓ Could not determine sentiment")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            demo.save_session_data()
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"❌ An error occurred: {e}")


def main():
    print("🎬 Enhanced Sentiment Analysis Demo")
    print("=" * 45)
    
    while True:
        try:
            print("\nChoose an option:")
            print("1. Enhanced Interactive mode")
            print("2. Test sample reviews")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                enhanced_interactive_mode()
            elif choice == '2':
                try:
                    demo = EnhancedSentimentDemo()
                    demo.test_sample_reviews()
                except Exception as e:
                    print(f"❌ Error: {e}")
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main menu: {e}")
            print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()