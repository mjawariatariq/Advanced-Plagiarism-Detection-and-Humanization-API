
# fine_tuner.py

"""
Production-level Fine-Tuning for Plagiarism Detection
Features: PAN Dataset support, hyperparameter tuning, early stopping, best model saving
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed
)

warnings.filterwarnings('ignore')
set_seed(42)


class ProjectFineTuner:
    """Production-ready fine-tuner for Longformer plagiarism-detection model."""

    def __init__(self):
        self.model_name = "jpwahle/longformer-base-plagiarism-detection"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.max_length = 256
        self.best_metrics = {}
        self.training_history = []

    # ------------------------------------------------------------------ #
    # Data Loading & PAN Dataset Support
    # ------------------------------------------------------------------ #

    def load_pan_dataset(self, pan_dir: str = "data/pan-plagiarism-corpus-2011") -> pd.DataFrame:
        """
        Load PAN 2011 Plagiarism Corpus from disk.
        Handles both XML and plain text files automatically.
        """
        print(f"\n📂 Scanning PAN dataset directory: {pan_dir}")
        
        texts = []
        labels = []
        
        # Check for PAN directory structure
        if not os.path.exists(pan_dir):
            print(f"⚠️ PAN directory not found: {pan_dir}")
            return pd.DataFrame()
        
        # Look for source-document and suspicious-document directories
        source_dir = os.path.join(pan_dir, "source-document")
        suspicious_dir = os.path.join(pan_dir, "suspicious-document")
        
        # If standard PAN structure exists
        if os.path.exists(source_dir) and os.path.exists(suspicious_dir):
            print("✅ Found standard PAN 2011 directory structure")
            
            # Load source documents (original - label 0)
            source_count = self._load_documents_from_dir(source_dir, texts, labels, label=0)
            print(f"   Loaded {source_count} original documents")
            
            # Load suspicious documents (plagiarized - label 1)
            suspicious_count = self._load_documents_from_dir(suspicious_dir, texts, labels, label=1)
            print(f"   Loaded {suspicious_count} plagiarized documents")
        
        else:
            # Fallback: Try to load any .txt files from the directory
            print("⚠️ Standard PAN structure not found. Scanning for .txt files...")
            self._load_all_text_files(pan_dir, texts, labels)
        
        if not texts:
            print("❌ No documents found in PAN dataset")
            return pd.DataFrame()
        
        print(f"✅ Loaded {len(texts)} samples from PAN dataset")
        print(f"   Original: {labels.count(0)}")
        print(f"   Plagiarized: {labels.count(1)}")
        
        return pd.DataFrame({'text': texts, 'label': labels})

    def _load_documents_from_dir(self, directory: str, texts: List, labels: List, label: int) -> int:
        """Load all text files from a directory."""
        count = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                            if content and len(content) > 50:  # Minimum content length
                                texts.append(content)
                                labels.append(label)
                                count += 1
                    except Exception as e:
                        print(f"   ⚠️ Error reading {file_path}: {e}")
        return count

    def _load_all_text_files(self, directory: str, texts: List, labels: List):
        """Fallback: Load all .txt files and guess labels based on filename patterns."""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                            if content and len(content) > 50:
                                texts.append(content)
                                # Guess label based on filename
                                if 'suspicious' in file.lower() or 'plagiarized' in file.lower():
                                    labels.append(1)
                                elif 'source' in file.lower() or 'original' in file.lower():
                                    labels.append(0)
                                else:
                                    labels.append(1)  # Default to plagiarized
                    except Exception:
                        pass

    def load_dataset(self, data_path: str = 'data/plagiarism_dataset.csv') -> pd.DataFrame:
        """Load dataset from CSV, JSON, or PAN directory."""
        print(f"\n📂 Loading dataset...")
        
        # Check if it's a PAN directory
        if os.path.isdir(data_path):
            return self.load_pan_dataset(data_path)
        
        # Try loading as CSV/JSON
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                # Try guessing format
                with open(data_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('[') or first_line.startswith('{'):
                        df = pd.read_json(data_path)
                    else:
                        df = pd.read_csv(data_path)
            
            # IMPORTANT: Ensure text column is string type
            if 'text' in df.columns:
                df['text'] = df['text'].astype(str)
            
            print(f"   Loaded {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Dataset Preparation
    # ------------------------------------------------------------------ #

    def prepare_datasets(self, df: pd.DataFrame, test_size=0.2, val_size=0.1, balance_data=True):
        """
        Split DataFrame into train/val/test HF Datasets with optional balancing.
        """
        print("\n📊 Preparing datasets...")
        
        # Validate data
        if df.empty:
            raise ValueError("Empty dataset provided")
        
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns")
        
        # Convert text to string and handle NaN values
        df['text'] = df['text'].fillna('').astype(str)
        
        # Filter out empty texts
        df = df[df['text'].str.len() > 10]
        print(f"   After filtering: {len(df)} samples")
        
        # Balance dataset if needed
        if balance_data:
            df = self._balance_dataset(df)
            print(f"   Balanced dataset: {len(df)} samples")
        
        # Split into train/val/test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            df['text'].tolist(), df['label'].tolist(),
            test_size=test_size, random_state=42, stratify=df['label']
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_size_adjusted, random_state=42, stratify=train_val_labels
        )
        
        print(f"   Training: {len(train_texts)} samples")
        print(f"   Validation: {len(val_texts)} samples")
        print(f"   Test: {len(test_texts)} samples")
        
        # Create Hugging Face datasets
        train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
        val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
        test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
        
        return train_dataset, val_dataset, test_dataset

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance dataset by oversampling minority class."""
        label_counts = df['label'].value_counts()
        if len(label_counts) < 2:
            return df
        
        max_count = label_counts.max()
        
        # Oversample minority classes
        balanced_dfs = []
        for label in label_counts.index:
            class_df = df[df['label'] == label]
            if len(class_df) < max_count:
                # Oversample with replacement
                oversampled = class_df.sample(n=max_count, replace=True, random_state=42)
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Model Initialization
    # ------------------------------------------------------------------ #

    def initialize_model(self, resume_from_checkpoint: Optional[str] = None):
        """Initialize model from pretrained or checkpoint."""
        print("\n🔄 Initializing model...")
        
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            print(f"   Resuming from checkpoint: {resume_from_checkpoint}")
            self.tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                resume_from_checkpoint, num_labels=2, ignore_mismatched_sizes=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2, ignore_mismatched_sizes=True
            )
        
        self.model.to(self.device)
        print(f"✅ Model initialized on {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model

    def tokenize_datasets(self, train_dataset: Dataset, val_dataset: Dataset):
        """Tokenize datasets with proper padding and truncation."""
        print("\n🔄 Tokenizing datasets...")
        
        def tokenize_function(examples):
            # Ensure text is string
            texts = [str(text) for text in examples['text']]
            return self.tokenizer(
                texts,
                truncation=True, 
                padding='max_length', 
                max_length=self.max_length
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Remove the text column to avoid conflicts
        train_dataset = train_dataset.remove_columns(['text'])
        val_dataset = val_dataset.remove_columns(['text'])
        
        print("✅ Tokenization complete")
        return train_dataset, val_dataset

    # ------------------------------------------------------------------ #
    # Trainer Setup & Training
    # ------------------------------------------------------------------ #

    def setup_trainer(self, train_dataset: Dataset, val_dataset: Dataset,
                      output_dir: str = './models/project_model', 
                      epochs: int = 5, 
                      batch_size: int = 4,
                      learning_rate: float = 2e-5,
                      weight_decay: float = 0.01,
                      warmup_ratio: float = 0.1,
                      gradient_accumulation_steps: int = 4,
                      early_stopping_patience: int = 3,
                      load_best_model: bool = True):
        """
        Configure the Trainer with advanced hyperparameters.
        """
        print("\n⚙️ Setting up advanced trainer configuration...")
        
        def compute_metrics(eval_pred):
            """Compute all metrics including confusion matrix and ROC-AUC."""
            predictions, labels = eval_pred
            preds = np.argmax(predictions, axis=1)
            probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()[:, 1]
            
            # Calculate all metrics
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)
            precision = precision_score(labels, preds, average='weighted', zero_division=0)
            recall = recall_score(labels, preds, average='weighted', zero_division=0)
            
            # ROC-AUC (handles edge cases)
            try:
                roc_auc = roc_auc_score(labels, probs)
            except Exception:
                roc_auc = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(labels, preds).tolist()
            
            # Classification report
            report = classification_report(labels, preds, zero_division=0)
            
            print("\n📊 Evaluation Metrics:")
            print(f"   Accuracy: {acc:.4f}")
            print(f"   F1 Score: {f1:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   Confusion Matrix:\n{cm}")
            
            # Store metrics
            self.best_metrics = {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'confusion_matrix': cm
            }
            
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc
            }
        
        # Training arguments with advanced settings
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=load_best_model,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir='./logs',
            logging_steps=10,
            push_to_hub=False,
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            save_total_limit=3,
            remove_unused_columns=False,
            # label_names=['label']
        )
        
        # Create trainer with callbacks
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ]
        )
        
        print("✅ Trainer setup complete")
        return self.trainer

    def train(self, save_path: str = './models/fine_tuned_model') -> Dict:
        """Run training and save the best model."""
        if not self.trainer:
            raise ValueError("Trainer not initialized! Call setup_trainer() first.")
        
        print("\n" + "=" * 70)
        print("  🚀 STARTING PRODUCTION TRAINING")
        print("=" * 70)
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save training history
        self.training_history = {
            'train_loss': train_result.training_loss,
            'epochs_trained': train_result.metrics.get('epoch', 0),
            'train_runtime': train_result.metrics.get('train_runtime', 0)
        }
        
        # Evaluate on validation set
        eval_result = self.trainer.evaluate()
        
        # Save the best model
        print("\n💾 Saving best model...")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save metrics
        metrics_path = os.path.join(save_path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'training': self.training_history,
                'evaluation': eval_result,
                'best_metrics': self.best_metrics,
                'hyperparameters': {
                    'model_name': self.model_name,
                    'max_length': self.max_length,
                    'device': str(self.device)
                }
            }, f, indent=2)
        
        print(f"✅ Model saved to: {save_path}")
        print(f"✅ Metrics saved to: {metrics_path}")
        
        # Print final results
        print("\n" + "=" * 70)
        print("  📊 PRODUCTION TRAINING RESULTS")
        print("=" * 70)
        print(f"   Training Loss: {train_result.training_loss:.4f}")
        print(f"   Eval Accuracy: {eval_result.get('eval_accuracy', 0) * 100:.2f}%")
        print(f"   Eval F1 Score: {eval_result.get('eval_f1', 0):.4f}")
        print(f"   Eval Precision: {eval_result.get('eval_precision', 0):.4f}")
        print(f"   Eval Recall: {eval_result.get('eval_recall', 0):.4f}")
        print(f"   Eval ROC-AUC: {eval_result.get('eval_roc_auc', 0):.4f}")
        print("=" * 70)
        
        return {
            'train_loss': train_result.training_loss,
            'eval_metrics': eval_result,
            'model_path': save_path,
            'best_metrics': self.best_metrics,
            'training_history': self.training_history
        }

    # ------------------------------------------------------------------ #
    # Inference & Testing
    # ------------------------------------------------------------------ #

    def predict(self, text: str, model_path: str = None, confidence_threshold: float = 0.5) -> Dict:
        """
        Predict on a single text with confidence threshold.
        """
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded. Call initialize_model() or pass model_path.")
        
        self.model.eval()
        inputs = self.tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1)
        
        prob = probabilities.cpu().numpy()[0]
        is_plagiarized = bool(prediction.item() == 1)
        confidence = float(max(prob))
        
        # Apply confidence threshold
        if confidence < confidence_threshold:
            is_plagiarized = False
            confidence = 1 - confidence
        
        return {
            'is_plagiarized': is_plagiarized,
            'label': 'Plagiarized' if is_plagiarized else 'Original',
            'confidence': confidence,
            'probability': prob.tolist(),
            'probability_plagiarized': float(prob[1]),
            'probability_original': float(prob[0])
        }

    def test_model(self, test_dataset: Dataset, model_path: str = './models/fine_tuned_model'):
        """Run comprehensive test on the fine-tuned model."""
        print("\n" + "=" * 70)
        print("  🔍 COMPREHENSIVE MODEL TESTING")
        print("=" * 70)
        
        # Test with known examples
        test_texts = [
            # Original texts
            "Artificial intelligence is the simulation of human intelligence in machines.",
            "Machine learning enables computers to learn from data without explicit programming.",
            
            # Plagiarized texts
            "The simulation of human intelligence in machines is known as artificial intelligence.",
            "Computers can learn from data without programming through machine learning.",
            
            # Mixed
            "Deep learning uses neural networks with multiple layers to process complex patterns.",
            "Neural networks with multiple layers are used in deep learning."
        ]
        
        print("\n🔬 Single-sample predictions:")
        print("-" * 60)
        for i, text in enumerate(test_texts, 1):
            pred = self.predict(text, model_path=model_path if i == 0 else None)
            print(f"\nTest {i}:")
            print(f"Text: {text[:80]}...")
            print(f"Prediction: {pred['label']}")
            print(f"Confidence: {pred['confidence'] * 100:.1f}%")
            print(f"Probabilities: Original: {pred['probability_original']*100:.1f}%, Plagiarized: {pred['probability_plagiarized']*100:.1f}%")
        print("-" * 60)
        
        # Batch evaluation on test dataset
        if test_dataset:
            print("\n📊 Batch evaluation on test set:")
            predictions = self.trainer.predict(test_dataset)
            metrics = predictions.metrics
            
            print(f"   Accuracy: {metrics.get('test_accuracy', 0) * 100:.2f}%")
            print(f"   F1 Score: {metrics.get('test_f1', 0):.4f}")
            print(f"   Precision: {metrics.get('test_precision', 0):.4f}")
            print(f"   Recall: {metrics.get('test_recall', 0):.4f}")
            print(f"   ROC-AUC: {metrics.get('test_roc_auc', 0):.4f}")
        
        return self.best_metrics

    # ------------------------------------------------------------------ #
    # Utility Methods
    # ------------------------------------------------------------------ #

    def save_model_weights(self, path: str = './models/fine_tuned_model'):
        """Save only model weights (smaller file)."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'model_weights.pt'))
        print(f"✅ Model weights saved to {path}")

    def load_model_weights(self, path: str = './models/fine_tuned_model'):
        """Load only model weights."""
        weights_path = os.path.join(path, 'model_weights.pt')
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✅ Model weights loaded from {path}")
            return True
        return False

    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_loaded': self.model is not None,
            'is_trained': self.trainer is not None,
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'max_length': self.max_length,
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'available_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


# Backward compatibility
PlagiarismFineTuner = ProjectFineTuner