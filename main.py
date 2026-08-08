

# #!/usr/bin/env python3
# """
# Plagiarism Detection & AI Humanization System
# Hugging Face Longformer Integration with Gemini API
# Uses a fine-tuned model when available (falls back to the base model, ~81% acc).

# FIX (this version): detector.predict() now receives the RAW text instead of
# preprocessor.preprocess(text)['cleaned_text']. During training (fine_tuner.py),
# the model was tokenized straight from the raw 'text' column - no stopword
# removal, no lowercasing, no punctuation stripping. Feeding it stripped/cleaned
# text at inference time put every input out-of-distribution relative to what
# the model actually learned, which is why it was predicting "Plagiarized 100%"
# on virtually everything regardless of content. The preprocessor is still used
# for word-count/display stats, just no longer as model input.
# """

# import os
# import sys
# import warnings
# from datetime import datetime

# warnings.filterwarnings('ignore')

# current_dir = os.path.dirname(os.path.abspath(__file__))
# src_path = os.path.join(current_dir, 'src')
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)

# from src.preprocessing import TextPreprocessor
# from src.huggingface_detector import HuggingFaceDetector
# from src.text_humanizer import GeminiHumanizer

# try:
#     from dotenv import load_dotenv
#     if os.path.exists(".env"):
#         load_dotenv(dotenv_path=".env", encoding="utf-8")
# except ImportError:
#     print("⚠️ python-dotenv not installed, using system env")


# class PlagiarismApp:
#     """Main application: Hugging Face detector + Gemini humanization."""

#     def __init__(self):
#         self.preprocessor = TextPreprocessor()
#         self.detector = HuggingFaceDetector()
#         self.humanizer = GeminiHumanizer()
#         self.is_trained = False
#         self.last_results = None
#         self.session_history = []
#         # Uses the pretrained jpwahle/longformer-base-plagiarism-detection
#         # model directly - no training/fine-tuning dataset required. If a
#         # fine-tuned model exists at ./models/fine_tuned_model it will still
#         # be picked up automatically, but that is optional, not required.
#         self.use_fine_tuned = True

#     def train(self) -> bool:
#         """Load the model (fine-tuned if available, otherwise base)."""
#         print("\n" + "=" * 60)
#         print("🚀 LOADING MODEL")
#         print("=" * 60)

#         fine_tuned_path = "./models/fine_tuned_model"

#         if self.use_fine_tuned and os.path.exists(fine_tuned_path):
#             print(f"\n📂 Fine-tuned model found at: {fine_tuned_path}")
#             success = self.detector.load_fine_tuned_model(fine_tuned_path)
#             if success:
#                 self.is_trained = True
#                 self._test_model()
#                 return True

#         print("\n🔄 Loading base model...")
#         success = self.detector.load_model()
#         if success:
#             self.is_trained = True
#             self._test_model()
#             return True

#         print("\n❌ Failed to load model")
#         return False

#     def _test_model(self):
#         test_text = "Artificial intelligence is the simulation of human intelligence in machines."
#         result = self.detector.predict(test_text)
#         print(f"\n🔍 Test prediction: {result['label']} ({result['confidence']*100:.1f}%)")

#     def run_menu(self):
#         self._print_header()
#         print("\n🔄 Loading model automatically...")
#         self.train()

#         while True:
#             self._print_menu()
#             choice = input("\n📌 Select option (1-7): ").strip()

#             menu_actions = {
#                 '1': self.check_plagiarism,
#                 '2': self.compare_texts,
#                 '3': self.humanize_text,
#                 '4': self.full_pipeline,
#                 '5': self.show_performance,
#                 '6': self.show_statistics,
#                 '7': self.reload_model,
#                 '0': self.exit_app
#             }

#             action = menu_actions.get(choice)
#             if action:
#                 action()
#             else:
#                 print("\n❌ Invalid option. Please try again.")

#     def _print_header(self):
#         model_type = "Fine-Tuned (92.5% accuracy)" if self.use_fine_tuned else "Base Model"
#         print("\n" + "=" * 70)
#         print("  📚 HUMA PLAG - Plagiarism Detection & AI Humanization")
#         print("  🚀 Powered by Hugging Face Longformer")
#         print(f"  🎯 Model: {model_type}")
#         print("  🤖 Gemini API - gemini-2.5-flash")
#         print("=" * 70)
#         print(f"  📅 {datetime.now().strftime('%B %d, %Y %H:%M')}")
#         print(f"  💻 Device: {self.detector.device}")
#         print("=" * 70)

#     def _print_menu(self):
#         print("\n" + "=" * 70)
#         print("  MAIN MENU")
#         print("=" * 70)
#         print("  1.  🔍 Check Text for Plagiarism")
#         print("  2.  📊 Compare Two Texts")
#         print("  3.  ✍️  Humanize Text with Gemini")
#         print("  4.  🔄 Full Pipeline (Detect + Humanize)")
#         print("  5.  📈 Show Model Performance")
#         print("  6.  📊 Show Statistics")
#         print("  7.  🔄 Reload Model")
#         print("  0.  🚪 Exit")
#         print("=" * 70)

#         if self.detector.is_fine_tuned:
#             model_status = "✅ Fine-Tuned (92.5%)"
#         elif self.is_trained:
#             model_status = "✅ Base Model (80.99%)"
#         else:
#             model_status = "⚠️ Not Loaded"

#         print(f"  📊 Model: {model_status}")
#         print(f"  🤖 Gemini API: {'✅ Available' if self.humanizer.available else '❌ Not Available'}")
#         print("=" * 70)

#     def check_plagiarism(self):
#         print("\n" + "=" * 50)
#         print("  🔍 PLAGIARISM CHECK")
#         print("=" * 50)

#         text = input("\n📝 Enter text (min 10 characters):\n> ").strip()
#         if len(text) < 10:
#             print("❌ Please enter at least 10 characters")
#             return

#         try:
#             # NOTE: model gets the RAW text - same format it was trained on.
#             # `preprocessed` is only used below for display stats (word count
#             # before/after stopword removal etc.), never fed to the model.
#             preprocessed = self.preprocessor.preprocess(text)
#             result = self.detector.predict(text)
#             semantic_sim = self.detector.calculate_semantic_similarity(text, text)

#             print("\n" + "=" * 50)
#             print("  📊 PLAGIARISM RESULTS")
#             print("=" * 50)
#             print(f"  📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
#             print(f"  📊 Word Count: {result.get('word_count', 0)}")
#             print(f"  🔍 Status: {result['label']}")
#             print(f"  📈 Confidence: {result['confidence']*100:.1f}%")
#             print(f"  ⚠️  Risk Level: {result.get('risk_level', 'Unknown')}")
#             print(f"  📊 Similarity Score: {result.get('similarity_score', 0)*100:.1f}%")
#             print(f"  🧠 Semantic Score (self): {semantic_sim*100:.1f}%")
#             print(f"  🎯 Model Type: {result.get('model_type', 'Unknown')}")

#             if result.get('probability'):
#                 print(f"  📊 Probability: [{result['probability'][0]*100:.1f}%, {result['probability'][1]*100:.1f}%]")

#             self.last_results = result
#             self.session_history.append({
#                 'type': 'plagiarism_check', 'text': text, 'result': result,
#                 'timestamp': datetime.now().isoformat()
#             })
#             print("=" * 50)

#         except Exception as e:
#             print(f"❌ Error: {e}")

#     def compare_texts(self):
#         print("\n" + "=" * 50)
#         print("  📊 COMPARE TWO TEXTS")
#         print("=" * 50)

#         text1 = input("\n📝 Enter first text:\n> ").strip()
#         text2 = input("\n📝 Enter second text:\n> ").strip()

#         if len(text1) < 5 or len(text2) < 5:
#             print("❌ Please enter at least 5 characters for each text")
#             return

#         try:
#             report = self.detector.compare_texts(text1, text2)

#             print("\n" + "=" * 50)
#             print("  📊 SIMILARITY REPORT")
#             print("=" * 50)
#             print(f"  📐 Cosine Similarity: {report.cosine_similarity*100:.2f}%")
#             print(f"  🧠 Semantic Similarity: {report.semantic_similarity*100:.2f}%")
#             print(f"  ⚠️  Risk Level: {report.risk_level}")
#             print(f"  📊 Risk Score: {report.risk_score*100:.1f}%")
#             print(f"  📅 Timestamp: {report.timestamp}")
#             print("=" * 50)

#         except Exception as e:
#             print(f"❌ Error: {e}")

#     def humanize_text(self):
#         print("\n" + "=" * 50)
#         print("  ✍️  TEXT HUMANIZATION (Gemini)")
#         print("=" * 50)

#         if not self.humanizer.available:
#             print("\n❌ Gemini API not available!")
#             print("💡 Please set GEMINI_API_KEY in .env file")
#             return

#         text = input("\n📝 Enter text to humanize (min 10 characters):\n> ").strip()
#         if len(text) < 10:
#             print("❌ Please enter at least 10 characters")
#             return

#         print("\n🎨 Select style:")
#         print("  1. Academic (formal)")
#         print("  2. Casual (conversational)")
#         print("  3. Professional (business)")
#         print("  4. Creative (engaging)")
#         print("  5. Simple (readable)")

#         style_choice = input("\nSelect style (1-5): ").strip()
#         style_map = {'1': 'academic', '2': 'casual', '3': 'professional', '4': 'creative', '5': 'simple'}
#         style = style_map.get(style_choice, 'academic')

#         try:
#             print(f"\n🔄 Humanizing with style: {style}")
#             result = self.humanizer.humanize(
#                 text=text, style=style,
#                 similarity_calculator=self.detector.calculate_semantic_similarity
#             )
#             self._display_humanization_result(result)
#         except Exception as e:
#             print(f"❌ Error: {e}")

#     def _display_humanization_result(self, result):
#         if result.get('success', False):
#             print("\n" + "=" * 50)
#             print("  ✍️  HUMANIZED TEXT")
#             print("=" * 50)
#             print(result['humanized_text'])
#             print("\n" + "=" * 50)
#             print("  📊 METRICS")
#             print("=" * 50)
#             print(f"  🤖 API Used: {result.get('api_used', 'Google Gemini')}")
#             print(f"  📝 Model: {result.get('model_used', 'gemini-2.5-flash')}")
#             print(f"  📝 Word count: {result.get('word_count', 0)}")
#             print(f"  ⏱️  Response time: {result.get('response_time', 0):.2f}s")
#             print(f"  🎨 Style: {result.get('style', 'Unknown')}")

#             if 'similarity_to_original' in result:
#                 print(f"\n  📊 Similarity Metrics:")
#                 print(f"     📐 Similarity to original: {result['similarity_to_original']*100:.1f}%")
#                 if 'similarity_change' in result:
#                     print(f"     📊 Similarity change: {result['similarity_change']*100:+.1f}%")
#                 if 'humanization_effective' in result:
#                     print(f"     ✅ Effective: {'Yes' if result['humanization_effective'] else 'No'}")

#             print("=" * 50)

#             self.session_history.append({
#                 'type': 'humanization',
#                 'original_text': result.get('original_text', ''),
#                 'humanized_text': result.get('humanized_text', ''),
#                 'result': result,
#                 'timestamp': datetime.now().isoformat()
#             })
#         else:
#             print(f"\n❌ Humanization failed: {result.get('error', 'Unknown error')}")

#     def full_pipeline(self):
#         print("\n" + "=" * 50)
#         print("  🔄 FULL PIPELINE")
#         print("=" * 50)

#         text = input("\n📝 Enter text:\n> ").strip()
#         if len(text) < 10:
#             print("❌ Please enter at least 10 characters")
#             return

#         try:
#             print("\n" + "=" * 50)
#             print("  STEP 1: PLAGIARISM DETECTION")
#             print("=" * 50)

#             # Same fix as check_plagiarism(): raw text goes to the model,
#             # not the stopword/punctuation-stripped version.
#             detection = self.detector.predict(text)

#             print(f"  🔍 Status: {detection['label']}")
#             print(f"  📈 Confidence: {detection['confidence']*100:.1f}%")
#             print(f"  ⚠️  Risk Level: {detection.get('risk_level', 'Unknown')}")
#             print(f"  📊 Similarity: {detection.get('similarity_score', 0)*100:.1f}%")
#             print(f"  🎯 Model: {detection.get('model_type', 'Unknown')}")

#             # Always call Gemini here - humanization runs on every text,
#             # not just when plagiarism is detected.
#             if self.humanizer.available:
#                 print("\n" + "=" * 50)
#                 print("  STEP 2: HUMANIZATION (Gemini - always runs)")
#                 print("=" * 50)

#                 humanized = self.humanizer.humanize(
#                     text=text, style='academic',
#                     similarity_calculator=self.detector.calculate_semantic_similarity
#                 )

#                 if humanized.get('success', False):
#                     print("\n✅ Humanization Complete!")
#                     print("\n" + "=" * 50)
#                     print("  ✍️  HUMANIZED TEXT")
#                     print("=" * 50)
#                     print(humanized['humanized_text'])
#                     print("\n" + "=" * 50)
#                     print("  📊 HUMANIZATION METRICS")
#                     print("=" * 50)
#                     print(f"  🤖 API: {humanized.get('api_used', 'Google Gemini')}")
#                     print(f"  📐 Similarity to original: {humanized.get('similarity_to_original', 0)*100:.1f}%")
#                     if 'similarity_change' in humanized:
#                         print(f"  📊 Similarity change: {humanized['similarity_change']*100:+.1f}%")
#                     print("=" * 50)
#                 else:
#                     print(f"\n❌ Humanization failed: {humanized.get('error', 'Unknown error')}")
#             else:
#                 print("\n⚠️  Gemini API not available for humanization")

#             self.session_history.append({
#                 'type': 'full_pipeline', 'text': text, 'detection': detection,
#                 'timestamp': datetime.now().isoformat()
#             })

#         except Exception as e:
#             print(f"❌ Error: {e}")

#     def show_performance(self):
#         print("\n" + "=" * 50)
#         print("  📈 MODEL PERFORMANCE")
#         print("=" * 50)

#         if not self.detector.is_loaded:
#             print("  ❌ Model not loaded")
#             return

#         metrics = self.detector.get_performance_metrics()

#         print(f"  🤖 Model: {metrics.get('model_name', 'Unknown')}")
#         print(f"  🎯 Type: {'Fine-Tuned (92.5%)' if self.detector.is_fine_tuned else 'Base Model'}")
#         print(f"  💻 Device: {metrics.get('device', 'Unknown')}")
#         print(f"  📊 Status: {'✅ Loaded' if metrics.get('is_loaded') else '❌ Not loaded'}")
#         print(f"  📊 Parameters: {metrics.get('model_parameters', 0):,}")
#         print(f"  📝 Total Predictions: {metrics.get('total_predictions', 0)}")
#         print(f"  ⏱️  Avg Inference Time: {metrics.get('avg_inference_time', 0)*1000:.2f}ms")

#         if self.detector.results:
#             print(f"\n  📊 Evaluation Metrics:")
#             for name, result in self.detector.results.items():
#                 print(f"  📈 {name}:")
#                 for key in ('accuracy', 'f1_score', 'precision', 'recall'):
#                     if key in result:
#                         print(f"     {key.replace('_', ' ').title()}: {result[key]:.4f}")

#         if self.detector.is_fine_tuned:
#             print(f"\n  🎯 Fine-Tuned Model Performance:")
#             print(f"     ✅ Validation Accuracy: 92.50%")
#             print(f"     ✅ F1 Score: 0.9246")
#             print(f"     ✅ Precision: 0.9348")
#             print(f"     ✅ Recall: 0.9250")

#         print("=" * 50)

#     def show_statistics(self):
#         print("\n" + "=" * 50)
#         print("  📊 SESSION STATISTICS")
#         print("=" * 50)

#         total_ops = len(self.session_history)
#         if total_ops == 0:
#             print("  📭 No operations performed yet")
#             print("=" * 50)
#             return

#         op_counts = {}
#         for entry in self.session_history:
#             op_type = entry.get('type', 'unknown')
#             op_counts[op_type] = op_counts.get(op_type, 0) + 1

#         print(f"  📝 Total Operations: {total_ops}")
#         print(f"  📊 Operation Breakdown:")
#         for op_type, count in op_counts.items():
#             print(f"     {op_type}: {count}")

#         humanizer_stats = self.humanizer.get_statistics()
#         if humanizer_stats['total_humanizations'] > 0:
#             print(f"\n  ✍️  Humanization Stats:")
#             print(f"     Total: {humanizer_stats['total_humanizations']}")
#             print(f"     Success Rate: {humanizer_stats['success_rate']:.1f}%")
#             print(f"     Avg Response Time: {humanizer_stats['avg_response_time']:.2f}s")
#             if humanizer_stats.get('api_usage'):
#                 print(f"     API Usage:")
#                 for api, count in humanizer_stats['api_usage'].items():
#                     print(f"        {api}: {count}")
#             if humanizer_stats.get('styles_used'):
#                 print(f"     Styles Used: {', '.join(humanizer_stats['styles_used'])}")

#         if self.detector.is_loaded:
#             print(f"\n  🎯 Model Info:")
#             print(f"     Type: {'Fine-Tuned' if self.detector.is_fine_tuned else 'Base'}")
#             print(f"     Predictions: {self.detector.performance_metrics.get('total_predictions', 0)}")

#         print("=" * 50)

#     def reload_model(self):
#         print("\n🔄 Reloading model...")
#         self.is_trained = False
#         self.detector.is_loaded = False
#         self.detector.is_fine_tuned = False
#         self.train()

#     def exit_app(self):
#         print("\n" + "=" * 50)
#         print("  👋 Thank you for using HUMA PLAG!")
#         print("  📊 Session Summary:")
#         total_ops = len(self.session_history)
#         print(f"  📝 Total Operations: {total_ops}")
#         if total_ops > 0:
#             print(f"  📅 Started: {self.session_history[0]['timestamp']}")
#             print(f"  📅 Ended: {datetime.now().isoformat()}")

#         if self.detector.is_fine_tuned:
#             print(f"  🎯 Model: Fine-Tuned (92.5% accuracy)")
#         elif self.detector.is_loaded:
#             print(f"  🎯 Model: Base Model (80.99% accuracy)")

#         print("=" * 50)
#         sys.exit(0)


# def main():
#     try:
#         app = PlagiarismApp()
#         app.run_menu()
#     except KeyboardInterrupt:
#         print("\n\n👋 Goodbye!")
#         sys.exit(0)
#     except Exception as e:
#         print(f"\n❌ Fatal error: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Plagiarism Detection & AI Humanization System
Hugging Face Longformer Integration with Gemini API
Uses a fine-tuned model when available (falls back to the base model, ~81% acc).

FIX (this version): detector.predict() now receives the RAW text instead of
preprocessor.preprocess(text)['cleaned_text']. During training (fine_tuner.py),
the model was tokenized straight from the raw 'text' column - no stopword
removal, no lowercasing, no punctuation stripping. Feeding it stripped/cleaned
text at inference time put every input out-of-distribution relative to what
the model actually learned, which is why it was predicting "Plagiarized 100%"
on virtually everything regardless of content. The preprocessor is still used
for word-count/display stats, just no longer as model input.
"""

import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.preprocessing import TextPreprocessor
from src.huggingface_detector import HuggingFaceDetector
from src.text_humanizer import GeminiHumanizer
from src.reference_corpus import load_reference_corpus

try:
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv(dotenv_path=".env", encoding="utf-8")
except ImportError:
    print("⚠️ python-dotenv not installed, using system env")


class PlagiarismApp:
    """Main application: Hugging Face detector + Gemini humanization."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.detector = HuggingFaceDetector()
        self.humanizer = GeminiHumanizer()
        # Loaded from data/reference_corpus.txt if present, else a tiny
        # built-in placeholder. Used to ground "Plagiarized" verdicts
        # against real source documents instead of trusting the
        # classifier's own style-based guess blindly - see
        # predict_grounded() in huggingface_detector.py.
        self.reference_texts = load_reference_corpus()
        self.is_trained = False
        self.last_results = None
        self.session_history = []
        # Uses the pretrained jpwahle/longformer-base-plagiarism-detection
        # model directly - no training/fine-tuning dataset required. If a
        # fine-tuned model exists at ./models/fine_tuned_model it will still
        # be picked up automatically, but that is optional, not required.
        self.use_fine_tuned = True

    def train(self) -> bool:
        """Load the model (fine-tuned if available, otherwise base)."""
        print("\n" + "=" * 60)
        print("🚀 LOADING MODEL")
        print("=" * 60)

        fine_tuned_path = "./models/fine_tuned_model"

        if self.use_fine_tuned and os.path.exists(fine_tuned_path):
            print(f"\n📂 Fine-tuned model found at: {fine_tuned_path}")
            success = self.detector.load_fine_tuned_model(fine_tuned_path)
            if success:
                self.is_trained = True
                self._test_model()
                return True

        print("\n🔄 Loading base model...")
        success = self.detector.load_model()
        if success:
            self.is_trained = True
            self._test_model()
            return True

        print("\n❌ Failed to load model")
        return False

    def _test_model(self):
        test_text = "Artificial intelligence is the simulation of human intelligence in machines."
        result = self.detector.predict(test_text)
        print(f"\n🔍 Test prediction: {result['label']} ({result['confidence']*100:.1f}%)")

    def run_menu(self):
        self._print_header()
        print("\n🔄 Loading model automatically...")
        self.train()

        while True:
            self._print_menu()
            choice = input("\n📌 Select option (1-7): ").strip()

            menu_actions = {
                '1': self.check_plagiarism,
                '2': self.compare_texts,
                '3': self.humanize_text,
                '4': self.full_pipeline,
                '5': self.show_performance,
                '6': self.show_statistics,
                '7': self.reload_model,
                '0': self.exit_app
            }

            action = menu_actions.get(choice)
            if action:
                action()
            else:
                print("\n❌ Invalid option. Please try again.")

    def _print_header(self):
        model_type = "Fine-Tuned (92.5% accuracy)" if self.use_fine_tuned else "Base Model"
        print("\n" + "=" * 70)
        print("  📚 HUMA PLAG - Plagiarism Detection & AI Humanization")
        print("  🚀 Powered by Hugging Face Longformer")
        print(f"  🎯 Model: {model_type}")
        print("  🤖 Gemini API - gemini-2.5-flash")
        print("=" * 70)
        print(f"  📅 {datetime.now().strftime('%B %d, %Y %H:%M')}")
        print(f"  💻 Device: {self.detector.device}")
        print("=" * 70)

    def _print_menu(self):
        print("\n" + "=" * 70)
        print("  MAIN MENU")
        print("=" * 70)
        print("  1.  🔍 Check Text for Plagiarism")
        print("  2.  📊 Compare Two Texts")
        print("  3.  ✍️  Humanize Text with Gemini")
        print("  4.  🔄 Full Pipeline (Detect + Humanize)")
        print("  5.  📈 Show Model Performance")
        print("  6.  📊 Show Statistics")
        print("  7.  🔄 Reload Model")
        print("  0.  🚪 Exit")
        print("=" * 70)

        if self.detector.is_fine_tuned:
            model_status = "✅ Fine-Tuned (92.5%)"
        elif self.is_trained:
            model_status = "✅ Base Model (80.99%)"
        else:
            model_status = "⚠️ Not Loaded"

        print(f"  📊 Model: {model_status}")
        print(f"  🤖 Gemini API: {'✅ Available' if self.humanizer.available else '❌ Not Available'}")
        print("=" * 70)

    def check_plagiarism(self):
        print("\n" + "=" * 50)
        print("  🔍 PLAGIARISM CHECK")
        print("=" * 50)

        text = input("\n📝 Enter text (min 10 characters):\n> ").strip()
        if len(text) < 10:
            print("❌ Please enter at least 10 characters")
            return

        try:
            # NOTE: model gets the RAW text - same format it was trained on.
            # `preprocessed` is only used below for display stats (word count
            # before/after stopword removal etc.), never fed to the model.
            preprocessed = self.preprocessor.preprocess(text)
            # predict_grounded(): only lets "Plagiarized" stand if a real
            # document in the reference corpus actually matches - see
            # predict_grounded() in huggingface_detector.py for why.
            result = self.detector.predict_grounded(text, self.reference_texts)
            semantic_sim = self.detector.calculate_semantic_similarity(text, text)

            print("\n" + "=" * 50)
            print("  📊 PLAGIARISM RESULTS")
            print("=" * 50)
            print(f"  📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"  📊 Word Count: {result.get('word_count', 0)}")
            print(f"  🔍 Status: {result['label']}")
            print(f"  📈 Confidence: {result['confidence']*100:.1f}%")
            print(f"  ⚠️  Risk Level: {result.get('risk_level', 'Unknown')}")
            print(f"  📊 Similarity Score: {result.get('similarity_score', 0)*100:.1f}%")
            print(f"  🧠 Semantic Score (self): {semantic_sim*100:.1f}%")
            print(f"  🎯 Model Type: {result.get('model_type', 'Unknown')}")

            if result.get('probability'):
                print(f"  📊 Probability: [{result['probability'][0]*100:.1f}%, {result['probability'][1]*100:.1f}%]")

            if result.get('reference_similarity', 0) > 0 or result.get('best_reference_match'):
                print(f"  📚 Reference Match: {result['reference_similarity']*100:.1f}% similarity")
                if result.get('best_reference_match'):
                    print(f"     Closest source: \"{result['best_reference_match']}\"")
                if not result.get('grounded', False) and 'Style-Flagged' in result.get('label', ''):
                    print("     ⚠️  No real source matched closely - verdict downgraded from Plagiarized.")

            self.last_results = result
            self.session_history.append({
                'type': 'plagiarism_check', 'text': text, 'result': result,
                'timestamp': datetime.now().isoformat()
            })
            print("=" * 50)

        except Exception as e:
            print(f"❌ Error: {e}")

    def compare_texts(self):
        print("\n" + "=" * 50)
        print("  📊 COMPARE TWO TEXTS")
        print("=" * 50)

        text1 = input("\n📝 Enter first text:\n> ").strip()
        text2 = input("\n📝 Enter second text:\n> ").strip()

        if len(text1) < 5 or len(text2) < 5:
            print("❌ Please enter at least 5 characters for each text")
            return

        try:
            report = self.detector.compare_texts(text1, text2)

            print("\n" + "=" * 50)
            print("  📊 SIMILARITY REPORT")
            print("=" * 50)
            print(f"  📐 Cosine Similarity: {report.cosine_similarity*100:.2f}%")
            print(f"  🧠 Semantic Similarity: {report.semantic_similarity*100:.2f}%")
            print(f"  ⚠️  Risk Level: {report.risk_level}")
            print(f"  📊 Risk Score: {report.risk_score*100:.1f}%")
            print(f"  📅 Timestamp: {report.timestamp}")
            print("=" * 50)

        except Exception as e:
            print(f"❌ Error: {e}")

    def humanize_text(self):
        print("\n" + "=" * 50)
        print("  ✍️  TEXT HUMANIZATION (Gemini)")
        print("=" * 50)

        if not self.humanizer.available:
            print("\n❌ Gemini API not available!")
            print("💡 Please set GEMINI_API_KEY in .env file")
            return

        text = input("\n📝 Enter text to humanize (min 10 characters):\n> ").strip()
        if len(text) < 10:
            print("❌ Please enter at least 10 characters")
            return

        print("\n🎨 Select style:")
        print("  1. Academic (formal)")
        print("  2. Casual (conversational)")
        print("  3. Professional (business)")
        print("  4. Creative (engaging)")
        print("  5. Simple (readable)")

        style_choice = input("\nSelect style (1-5): ").strip()
        style_map = {'1': 'academic', '2': 'casual', '3': 'professional', '4': 'creative', '5': 'simple'}
        style = style_map.get(style_choice, 'academic')

        try:
            print(f"\n🔄 Humanizing with style: {style}")
            result = self.humanizer.humanize(
                text=text, style=style,
                similarity_calculator=self.detector.calculate_semantic_similarity
            )
            self._display_humanization_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")

    def _display_humanization_result(self, result):
        if result.get('success', False):
            print("\n" + "=" * 50)
            print("  ✍️  HUMANIZED TEXT")
            print("=" * 50)
            print(result['humanized_text'])
            print("\n" + "=" * 50)
            print("  📊 METRICS")
            print("=" * 50)
            print(f"  🤖 API Used: {result.get('api_used', 'Google Gemini')}")
            print(f"  📝 Model: {result.get('model_used', 'gemini-2.5-flash')}")
            print(f"  📝 Word count: {result.get('word_count', 0)}")
            print(f"  ⏱️  Response time: {result.get('response_time', 0):.2f}s")
            print(f"  🎨 Style: {result.get('style', 'Unknown')}")

            if 'similarity_to_original' in result:
                print(f"\n  📊 Similarity Metrics:")
                print(f"     📐 Similarity to original: {result['similarity_to_original']*100:.1f}%")
                if 'similarity_change' in result:
                    print(f"     📊 Similarity change: {result['similarity_change']*100:+.1f}%")
                if 'humanization_effective' in result:
                    print(f"     ✅ Effective: {'Yes' if result['humanization_effective'] else 'No'}")

            print("=" * 50)

            self.session_history.append({
                'type': 'humanization',
                'original_text': result.get('original_text', ''),
                'humanized_text': result.get('humanized_text', ''),
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        else:
            print(f"\n❌ Humanization failed: {result.get('error', 'Unknown error')}")

    def full_pipeline(self):
        print("\n" + "=" * 50)
        print("  🔄 FULL PIPELINE")
        print("=" * 50)

        text = input("\n📝 Enter text:\n> ").strip()
        if len(text) < 10:
            print("❌ Please enter at least 10 characters")
            return

        try:
            print("\n" + "=" * 50)
            print("  STEP 1: PLAGIARISM DETECTION")
            print("=" * 50)

            # Same fix as check_plagiarism(): raw text goes to the model,
            # not the stopword/punctuation-stripped version. Also
            # grounded against the reference corpus, same as
            # check_plagiarism() - see predict_grounded().
            detection = self.detector.predict_grounded(text, self.reference_texts)

            print(f"  🔍 Status: {detection['label']}")
            print(f"  📈 Confidence: {detection['confidence']*100:.1f}%")
            print(f"  ⚠️  Risk Level: {detection.get('risk_level', 'Unknown')}")
            print(f"  📊 Similarity: {detection.get('similarity_score', 0)*100:.1f}%")
            print(f"  🎯 Model: {detection.get('model_type', 'Unknown')}")
            if detection.get('best_reference_match'):
                print(f"  📚 Reference Match: {detection['reference_similarity']*100:.1f}% - "
                      f"\"{detection['best_reference_match']}\"")

            # Always call Gemini here - humanization runs on every text,
            # not just when plagiarism is detected.
            if self.humanizer.available:
                print("\n" + "=" * 50)
                print("  STEP 2: HUMANIZATION (Gemini - always runs)")
                print("=" * 50)

                humanized = self.humanizer.humanize(
                    text=text, style='academic',
                    similarity_calculator=self.detector.calculate_semantic_similarity
                )

                if humanized.get('success', False):
                    print("\n✅ Humanization Complete!")
                    print("\n" + "=" * 50)
                    print("  ✍️  HUMANIZED TEXT")
                    print("=" * 50)
                    print(humanized['humanized_text'])
                    print("\n" + "=" * 50)
                    print("  📊 HUMANIZATION METRICS")
                    print("=" * 50)
                    print(f"  🤖 API: {humanized.get('api_used', 'Google Gemini')}")
                    print(f"  📐 Similarity to original: {humanized.get('similarity_to_original', 0)*100:.1f}%")
                    if 'similarity_change' in humanized:
                        print(f"  📊 Similarity change: {humanized['similarity_change']*100:+.1f}%")
                    print("=" * 50)
                else:
                    print(f"\n❌ Humanization failed: {humanized.get('error', 'Unknown error')}")
            else:
                print("\n⚠️  Gemini API not available for humanization")

            self.session_history.append({
                'type': 'full_pipeline', 'text': text, 'detection': detection,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            print(f"❌ Error: {e}")

    def show_performance(self):
        print("\n" + "=" * 50)
        print("  📈 MODEL PERFORMANCE")
        print("=" * 50)

        if not self.detector.is_loaded:
            print("  ❌ Model not loaded")
            return

        metrics = self.detector.get_performance_metrics()

        print(f"  🤖 Model: {metrics.get('model_name', 'Unknown')}")
        print(f"  🎯 Type: {'Fine-Tuned (92.5%)' if self.detector.is_fine_tuned else 'Base Model'}")
        print(f"  💻 Device: {metrics.get('device', 'Unknown')}")
        print(f"  📊 Status: {'✅ Loaded' if metrics.get('is_loaded') else '❌ Not loaded'}")
        print(f"  📊 Parameters: {metrics.get('model_parameters', 0):,}")
        print(f"  📝 Total Predictions: {metrics.get('total_predictions', 0)}")
        print(f"  ⏱️  Avg Inference Time: {metrics.get('avg_inference_time', 0)*1000:.2f}ms")

        if self.detector.results:
            print(f"\n  📊 Evaluation Metrics:")
            for name, result in self.detector.results.items():
                print(f"  📈 {name}:")
                for key in ('accuracy', 'f1_score', 'precision', 'recall'):
                    if key in result:
                        print(f"     {key.replace('_', ' ').title()}: {result[key]:.4f}")

        if self.detector.is_fine_tuned:
            print(f"\n  🎯 Fine-Tuned Model Performance:")
            print(f"     ✅ Validation Accuracy: 92.50%")
            print(f"     ✅ F1 Score: 0.9246")
            print(f"     ✅ Precision: 0.9348")
            print(f"     ✅ Recall: 0.9250")

        print("=" * 50)

    def show_statistics(self):
        print("\n" + "=" * 50)
        print("  📊 SESSION STATISTICS")
        print("=" * 50)

        total_ops = len(self.session_history)
        if total_ops == 0:
            print("  📭 No operations performed yet")
            print("=" * 50)
            return

        op_counts = {}
        for entry in self.session_history:
            op_type = entry.get('type', 'unknown')
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        print(f"  📝 Total Operations: {total_ops}")
        print(f"  📊 Operation Breakdown:")
        for op_type, count in op_counts.items():
            print(f"     {op_type}: {count}")

        humanizer_stats = self.humanizer.get_statistics()
        if humanizer_stats['total_humanizations'] > 0:
            print(f"\n  ✍️  Humanization Stats:")
            print(f"     Total: {humanizer_stats['total_humanizations']}")
            print(f"     Success Rate: {humanizer_stats['success_rate']:.1f}%")
            print(f"     Avg Response Time: {humanizer_stats['avg_response_time']:.2f}s")
            if humanizer_stats.get('api_usage'):
                print(f"     API Usage:")
                for api, count in humanizer_stats['api_usage'].items():
                    print(f"        {api}: {count}")
            if humanizer_stats.get('styles_used'):
                print(f"     Styles Used: {', '.join(humanizer_stats['styles_used'])}")

        if self.detector.is_loaded:
            print(f"\n  🎯 Model Info:")
            print(f"     Type: {'Fine-Tuned' if self.detector.is_fine_tuned else 'Base'}")
            print(f"     Predictions: {self.detector.performance_metrics.get('total_predictions', 0)}")

        print("=" * 50)

    def reload_model(self):
        print("\n🔄 Reloading model...")
        self.is_trained = False
        self.detector.is_loaded = False
        self.detector.is_fine_tuned = False
        self.train()

    def exit_app(self):
        print("\n" + "=" * 50)
        print("  👋 Thank you for using HUMA PLAG!")
        print("  📊 Session Summary:")
        total_ops = len(self.session_history)
        print(f"  📝 Total Operations: {total_ops}")
        if total_ops > 0:
            print(f"  📅 Started: {self.session_history[0]['timestamp']}")
            print(f"  📅 Ended: {datetime.now().isoformat()}")

        if self.detector.is_fine_tuned:
            print(f"  🎯 Model: Fine-Tuned (92.5% accuracy)")
        elif self.detector.is_loaded:
            print(f"  🎯 Model: Base Model (80.99% accuracy)")

        print("=" * 50)
        sys.exit(0)


def main():
    try:
        app = PlagiarismApp()
        app.run_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()