

# dataset_creater.py


"""
Dataset creation for Plagiarism Detection (Binary Classification)
Produces 2 classes: original (0) and plagiarized (1)
"""

import os
import random
import pandas as pd


class PlagiarismDatasetCreator:
    """Create binary dataset for plagiarism detection."""

    def __init__(self):
        self.topics = {
            'artificial_intelligence': [
                "Artificial intelligence is the simulation of human intelligence in machines.",
                "AI systems can learn from experience, adapt to new inputs, and perform human-like tasks.",
                "Machine learning enables computers to learn from data without explicit programming.",
                "Deep learning uses neural networks with multiple layers to process complex patterns.",
                "Natural language processing helps computers understand and generate human language."
            ],
            'machine_learning': [
                "Machine learning algorithms build models based on sample data to make predictions.",
                "Supervised learning uses labeled data to train models for classification.",
                "Unsupervised learning finds hidden patterns in unlabeled data.",
                "Reinforcement learning trains agents through rewards and punishments.",
                "Neural networks are inspired by the biological structure of the brain."
            ],
            'data_science': [
                "Data science combines statistics, programming, and domain knowledge to extract insights.",
                "Data visualization helps communicate complex information effectively.",
                "Big data refers to extremely large datasets that require advanced processing.",
                "Data preprocessing is essential for improving model performance.",
                "Feature engineering creates new variables to improve model accuracy."
            ],
            'nlp': [
                "Natural language processing bridges the gap between human language and computers.",
                "Tokenization is the process of breaking text into individual words or tokens.",
                "Named entity recognition identifies and classifies named entities in text.",
                "Sentiment analysis determines the emotional tone behind words.",
                "Text summarization creates concise summaries of longer documents."
            ],
            'cybersecurity': [
                "Cybersecurity involves protecting systems, networks, and programs from digital attacks.",
                "Encryption is the process of converting information into a code to prevent unauthorized access.",
                "Firewalls are network security devices that monitor and filter incoming and outgoing network traffic.",
                "Malware is malicious software designed to damage, disrupt, or gain unauthorized access to a system.",
                "Phishing is a cyberattack that uses fraudulent emails to trick users into revealing sensitive information."
            ],
            'cloud_computing': [
                "Cloud computing is the delivery of computing services over the internet.",
                "Infrastructure as a Service (IaaS) provides virtualized computing resources.",
                "Platform as a Service (PaaS) offers a platform for developing and managing applications.",
                "Software as a Service (SaaS) delivers software applications over the internet.",
                "Hybrid cloud combines private and public cloud infrastructure for greater flexibility."
            ]
        }

        self.paraphrase_templates = [
            "In other words, {original}",
            "To put it differently, {original}",
            "Essentially, {original}",
            "This means that {original}",
            "It can be said that {original}"
        ]

    def create_original_texts(self, num_samples=1000):
        """Generate original texts (label 0)."""
        original_texts = []
        for _ in range(num_samples):
            topic = random.choice(list(self.topics.keys()))
            topic_texts = self.topics[topic]
            
            num_sentences = random.randint(2, 3)
            selected = random.sample(topic_texts, min(num_sentences, len(topic_texts)))
            text = " ".join(selected)
            
            if random.random() > 0.6:
                text += " This concept is fundamental to modern technology."
            
            original_texts.append(text)
        return original_texts

    def create_plagiarized_texts(self, original_texts, num_samples=1000):
        """Generate plagiarized versions (label 1)."""
        plagiarized_texts = []
        synonyms = {
            'intelligence': ['understanding', 'comprehension', 'cognition'],
            'learning': ['studying', 'acquiring knowledge', 'understanding'],
            'data': ['information', 'statistics', 'figures'],
            'processing': ['handling', 'analyzing', 'computing'],
            'network': ['system', 'structure', 'framework'],
            'model': ['framework', 'structure', 'system'],
            'analysis': ['examination', 'study', 'evaluation'],
            'security': ['safety', 'protection', 'defense']
        }

        for _ in range(num_samples):
            original = random.choice(original_texts)
            technique = random.choice(['synonym', 'restructure', 'paraphrase', 'shuffle'])

            if technique == 'synonym':
                words = original.split()
                for j, word in enumerate(words):
                    word_lower = word.lower().strip('.,!?')
                    if word_lower in synonyms and random.random() > 0.5:
                        words[j] = random.choice(synonyms[word_lower])
                plagiarized = ' '.join(words)

            elif technique == 'restructure':
                sentences = original.split('. ')
                if len(sentences) > 1:
                    random.shuffle(sentences)
                    plagiarized = '. '.join(sentences)
                else:
                    plagiarized = original

            elif technique == 'paraphrase':
                template = random.choice(self.paraphrase_templates)
                plagiarized = template.format(original=original)

            else:  # shuffle
                words = original.split()
                if len(words) > 5:
                    for _ in range(random.randint(1, 2)):
                        idx = random.randint(0, len(words) - 2)
                        words[idx], words[idx+1] = words[idx+1], words[idx]
                    plagiarized = ' '.join(words)
                else:
                    plagiarized = original

            plagiarized_texts.append(plagiarized)
        
        return plagiarized_texts

    def create_complete_dataset(self, num_samples=2000) -> pd.DataFrame:
        """Create balanced binary dataset."""
        print("📊 Creating binary dataset...")
        print("=" * 60)
        
        half = num_samples // 2
        print(f"Creating {half} original texts...")
        original = self.create_original_texts(half)
        
        print(f"Creating {half} plagiarized texts...")
        plagiarized = self.create_plagiarized_texts(original, half)
        
        texts = original + plagiarized
        labels = [0] * len(original) + [1] * len(plagiarized)
        
        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        print(f"\n✅ Dataset created successfully!")
        print(f"   Total samples: {len(df)}")
        print(f"   Original (0): {len(df[df['label'] == 0])}")
        print(f"   Plagiarized (1): {len(df[df['label'] == 1])}")
        
        return df


def save_dataset(df: pd.DataFrame, base_path='data'):
    """Save dataset to CSV and create train/test splits."""
    os.makedirs(base_path, exist_ok=True)

    csv_path = os.path.join(base_path, 'plagiarism_dataset_binary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to: {csv_path}")

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_path = os.path.join(base_path, 'train_binary.csv')
    test_path = os.path.join(base_path, 'test_binary.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Train set saved to: {train_path} (samples: {len(train_df)})")
    print(f"✅ Test set saved to: {test_path} (samples: {len(test_df)})")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  📚 BINARY PLAGIARISM DATASET CREATION")
    print("=" * 70)

    creator = PlagiarismDatasetCreator()
    df = creator.create_complete_dataset(2000)  # 2000 samples

    print("\n💾 Saving dataset...")
    save_dataset(df)

    print("\n📊 Sample data (first 5 rows):")
    print("=" * 60)
    for i in range(5):
        sample = df.iloc[i]
        print(f"\nSample {i + 1}:")
        print(f"Text: {sample['text'][:150]}...")
        print(f"Label: {'Original' if sample['label'] == 0 else 'Plagiarized'}")
        print("-" * 40)

    print("\n" + "=" * 70)
    print("✅ Dataset creation complete!")
    print("=" * 70)