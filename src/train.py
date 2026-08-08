# #!/usr/bin/env python3
# """
# Training script for fine-tuning the Longformer plagiarism-detection model.

# Run from the project root, e.g.:
#     python -m src.train --generate_data --epochs 3 --batch_size 2
#     python -m src.train --data_path data/plagiarism_dataset.csv --epochs 5
# """

# import argparse
# import os
# import sys
# import warnings

# import pandas as pd

# warnings.filterwarnings('ignore')

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

# from fine_tuner import ProjectFineTuner
# from src.data_preparation import PlagiarismDataGenerator


# def create_sample_data():
#     """Small built-in sample dataset for a quick smoke test."""
#     original_texts = [
#         "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
#         "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
#         "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher level features from raw input.",
#         "Natural language processing is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.",
#         "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.",
#         "Robotics is the interdisciplinary branch of engineering and science that includes mechanical engineering, electronics engineering, information engineering, computer science, and others.",
#         "Data science is an interdisciplinary field that uses scientific methods, algorithms, processes, and systems to extract knowledge and insights from structured and unstructured data.",
#         "The Internet of Things describes physical objects with sensors, processing ability, software, and other technologies that connect and exchange data with other devices and systems.",
#         "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
#         "Blockchain is a distributed ledger technology that maintains a growing list of records called blocks that are linked using cryptography."
#     ]

#     plagiarized_texts = [
#         "The simulation of human intelligence in machines that are programmed to think like humans and mimic their actions is known as artificial intelligence.",
#         "Systems that can learn and improve from experience without being explicitly programmed are enabled by machine learning, which is part of AI.",
#         "Using neural networks with multiple layers to extract features from raw input is called deep learning, a subset of machine learning.",
#         "NLP is a part of AI that helps computers to process and understand human language, enabling interaction between humans and machines.",
#         "Computer vision, a field of AI, enables computers to understand and interpret visual information from the world.",
#         "The multidisciplinary field combining mechanical engineering, electronics, and computer science to create intelligent machines is called robotics.",
#         "Data science combines scientific methods and algorithms to extract insights from both structured and unstructured data.",
#         "IoT connects physical devices with sensors and software to exchange data with other systems and devices.",
#         "Quantum computing uses quantum mechanics properties like superposition and entanglement to perform complex calculations.",
#         "Blockchain technology maintains a distributed ledger of records called blocks that are secured using cryptography."
#     ]

#     texts = original_texts + plagiarized_texts
#     labels = [0] * len(original_texts) + [1] * len(plagiarized_texts)
#     return texts, labels


# def main():
#     parser = argparse.ArgumentParser(description='Fine-tune Longformer for plagiarism detection')
#     parser.add_argument('--data_path', type=str, help='Path to dataset (CSV or JSON)')
#     parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
#     parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
#     parser.add_argument('--save_path', type=str, default='./models/fine_tuned_model',
#                          help='Path to save the fine-tuned model')
#     parser.add_argument('--generate_data', action='store_true',
#                          help='Generate synthetic data for training')
#     parser.add_argument('--num_samples', type=int, default=100,
#                          help='Number of synthetic samples to generate')
#     args = parser.parse_args()

#     print("\n" + "=" * 70)
#     print("  🎯 FINE-TUNE LONGFORMER - PLAGIARISM DETECTION")
#     print("=" * 70)

#     # --- Load or generate data -------------------------------------------------
#     if args.data_path:
#         if args.data_path.endswith('.csv'):
#             df = pd.read_csv(args.data_path)
#         elif args.data_path.endswith('.json'):
#             df = pd.read_json(args.data_path)
#         else:
#             print(f"❌ Unsupported file format: {args.data_path}")
#             print("   Supported formats: .csv, .json")
#             return
#         print(f"📂 Loaded {len(df)} samples from {args.data_path}")

#     elif args.generate_data:
#         print(f"\n📊 Generating {args.num_samples} synthetic samples...")
#         generator = PlagiarismDataGenerator()
#         texts, labels = generator.create_dataset(
#             num_original=args.num_samples // 2,
#             num_plagiarized=args.num_samples // 2
#         )
#         os.makedirs('data', exist_ok=True)
#         df = generator.save_dataset(texts, labels, 'data/generated_dataset.csv')
#         print(f"   Generated {len(df)} samples")

#     else:
#         print("\n📊 Using built-in sample dataset...")
#         texts, labels = create_sample_data()
#         df = pd.DataFrame({'text': texts, 'label': labels})
#         print(f"   Loaded {len(df)} samples")

#     if len(df) == 0:
#         print("❌ No data available for training")
#         return

#     if 'text' not in df.columns or 'label' not in df.columns:
#         print("❌ Dataset must contain 'text' and 'label' columns")
#         return

#     # --- Fine-tune ---------------------------------------------------------
#     print("\n🔧 Initializing fine-tuner...")
#     fine_tuner = ProjectFineTuner()

#     print("\n📊 Preparing dataset...")
#     train_dataset, val_dataset, test_dataset = fine_tuner.prepare_datasets(df)

#     print("\n🔄 Initializing Longformer model...")
#     fine_tuner.initialize_model()

#     print("\n🔄 Tokenizing datasets...")
#     train_dataset, val_dataset = fine_tuner.tokenize_datasets(train_dataset, val_dataset)

#     print("\n⚙️ Setting up trainer...")
#     fine_tuner.setup_trainer(
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         output_dir="./training_output",
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         learning_rate=args.learning_rate
#     )

#     print("\n" + "=" * 70)
#     print("  🚀 STARTING TRAINING")
#     print("=" * 70)

#     result = fine_tuner.train(save_path=args.save_path)

#     print("\n" + "=" * 70)
#     print("  🔍 TESTING THE FINE-TUNED MODEL")
#     print("=" * 70)
#     fine_tuner.test_model(test_dataset, model_path=args.save_path)

#     print("\n" + "=" * 70)
#     print("✅ Fine-tuning complete!")
#     print(f"   Model saved to: {result['model_path']}")
#     print(f"   Final training loss: {result.get('train_loss', 0):.4f}")

#     if 'eval_metrics' in result:
#         metrics = result['eval_metrics']
#         print(f"   Eval accuracy: {metrics.get('eval_accuracy', 0) * 100:.2f}%")
#         print(f"   Eval F1: {metrics.get('eval_f1', 0):.4f}")
#     print("=" * 70)

#     print("\n📝 To use this fine-tuned model in your application:")
#     print(f"   detector.load_fine_tuned_model('{args.save_path}')")

#     print("\n💡 To try different configurations:")
#     print("   python -m src.train --generate_data --epochs 5 --batch_size 4 --learning_rate 1e-5")
#     print("   python -m src.train --data_path data/your_dataset.csv --epochs 10")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Production-level training script with PAN dataset support,
advanced hyperparameters, and comprehensive evaluation.
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import torch

warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.fine_tuner import ProjectFineTuner


def main():
    parser = argparse.ArgumentParser(description='Production-level Fine-Tuning for Plagiarism Detection')
    
    # Data options
    parser.add_argument('--data_path', type=str, default='models/fine_tuned_model/',
                        help='Path to dataset (CSV/JSON) or PAN directory')
    parser.add_argument('--pan_dir', type=str, default='models/fine_tuned_model/',
                        help='Path to PAN 2011 corpus directory')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience')
    
    # Model options
    parser.add_argument('--save_path', type=str, default='./models/fine_tuned_model',
                        help='Path to save the fine-tuned model')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--test_only', action='store_true',
                        help='Only test existing model, no training')
    parser.add_argument('--eval_dataset', action='store_true',
                        help='Evaluate on test dataset after training')
    
    # Dataset options
    parser.add_argument('--balance_data', action='store_true', default=True,
                        help='Balance dataset by oversampling')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size ratio')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set size ratio')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  🚀 PRODUCTION-LEVEL PLAGIARISM DETECTION TRAINING")
    print("=" * 70)
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    # Initialize fine-tuner
    fine_tuner = ProjectFineTuner()
    
    # --- Load Dataset -------------------------------------------------
    print("\n📂 Loading dataset...")
    df = pd.DataFrame()
    
    # Try PAN dataset first
    if os.path.exists(args.pan_dir) and not args.data_path:
        print("   Using PAN 2011 Plagiarism Corpus...")
        df = fine_tuner.load_pan_dataset(args.pan_dir)
    
    # If PAN failed or explicitly specified, load from data_path
    if df.empty and args.data_path:
        print(f"   Loading from: {args.data_path}")
        df = fine_tuner.load_dataset(args.data_path)
    
    if df.empty:
        print("❌ No valid dataset found!")
        print("💡 Options:")
        print("   1. Download PAN 2011 corpus to data/pan-plagiarism-corpus-2011/")
        print("   2. Provide a CSV/JSON file with 'text' and 'label' columns")
        print("   3. Use --data_path with your dataset")
        sys.exit(1)
    
    # --- Test Mode ---------------------------------------------------
    if args.test_only:
        print("\n🔍 Test mode: Loading model and running tests...")
        fine_tuner.initialize_model()
        if os.path.exists(args.save_path):
            fine_tuner.model = fine_tuner.model.from_pretrained(args.save_path)
            fine_tuner.tokenizer = fine_tuner.tokenizer.from_pretrained(args.save_path)
            print(f"✅ Loaded model from: {args.save_path}")
        
        # Test on a few samples
        test_texts = [
            "Artificial intelligence is the simulation of human intelligence in machines.",
            "The simulation of human intelligence in machines is known as artificial intelligence."
        ]
        
        print("\n📊 Test predictions:")
        for text in test_texts:
            pred = fine_tuner.predict(text)
            print(f"   Text: {text[:50]}...")
            print(f"   Prediction: {pred['label']} ({pred['confidence']*100:.1f}%)")
        return
    
    # --- Prepare Datasets ---------------------------------------------
    train_dataset, val_dataset, test_dataset = fine_tuner.prepare_datasets(
        df, 
        test_size=args.test_size,
        val_size=args.val_size,
        balance_data=args.balance_data
    )
    
    # --- Initialize Model ---------------------------------------------
    print("\n🔄 Initializing model...")
    fine_tuner.initialize_model(resume_from_checkpoint=args.resume_from)
    
    # --- Tokenize ----------------------------------------------------
    train_dataset, val_dataset = fine_tuner.tokenize_datasets(train_dataset, val_dataset)
    
    # --- Setup Trainer ------------------------------------------------
    print("\n⚙️ Setting up trainer with optimized hyperparameters...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Weight Decay: {args.weight_decay}")
    print(f"   Warmup Ratio: {args.warmup_ratio}")
    print(f"   Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"   Early Stopping Patience: {args.early_stopping_patience}")
    
    fine_tuner.setup_trainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir='./training_output',
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        load_best_model=True
    )
    
    # --- Train -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  🚀 STARTING TRAINING...")
    print("=" * 70)
    
    result = fine_tuner.train(save_path=args.save_path)
    
    # --- Evaluate ----------------------------------------------------
    print("\n" + "=" * 70)
    print("  📊 FINAL EVALUATION")
    print("=" * 70)
    
    # Test on validation set
    val_predictions = fine_tuner.trainer.predict(val_dataset)
    print(f"\n✅ Validation Results:")
    for key, value in val_predictions.metrics.items():
        if key.startswith('test_'):
            print(f"   {key.replace('test_', '').title()}: {value:.4f}")
    
    # Test on test set if available
    if test_dataset and args.eval_dataset:
        test_predictions = fine_tuner.trainer.predict(test_dataset)
        print(f"\n✅ Test Results:")
        for key, value in test_predictions.metrics.items():
            if key.startswith('test_'):
                print(f"   {key.replace('test_', '').title()}: {value:.4f}")
    
    # --- Save Metrics ------------------------------------------------
    metrics_file = os.path.join(args.save_path, 'training_summary.txt')
    with open(metrics_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  PRODUCTION TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Model: {fine_tuner.model_name}\n")
        f.write(f"  Device: {fine_tuner.device}\n")
        f.write(f"  Save Path: {args.save_path}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("  HYPERPARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  Weight Decay: {args.weight_decay}\n")
        f.write(f"  Warmup Ratio: {args.warmup_ratio}\n")
        f.write(f"  Gradient Accumulation: {args.gradient_accumulation_steps}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("  FINAL METRICS\n")
        f.write("=" * 70 + "\n")
        for key, value in result['eval_metrics'].items():
            if key.startswith('eval_'):
                f.write(f"  {key.replace('eval_', '').title()}: {value:.4f}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n✅ Training summary saved to: {metrics_file}")
    
    print("\n" + "=" * 70)
    print("  ✅ TRAINING COMPLETE!")
    print("  🎯 Model ready for production use")
    print("=" * 70)
    
    print("\n📝 Next steps:")
    print("   1. Use the fine-tuned model in main.py")
    print("   2. Run API server with python api_integration.py")
    print("   3. The model will automatically load from ./models/fine_tuned_model")
    print("\n💡 Quick test:")
    print("   python -m src.train --test_only")


if __name__ == "__main__":
    main()