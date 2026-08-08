"""
Data Preparation for Plagiarism Detection Fine-Tuning
"""

import os
import random
from typing import List, Tuple

import pandas as pd


class PlagiarismDataGenerator:
    """Generate a synthetic plagiarism dataset for fine-tuning."""

    def __init__(self):
        self.base_texts = [
            "Machine learning algorithms enable computers to learn from data patterns.",
            "Artificial intelligence is rapidly transforming various industries worldwide.",
            "Deep learning models use neural networks for complex pattern recognition.",
            "Natural language processing helps computers understand human communication.",
            "Computer vision systems analyze and interpret visual information from images.",
            "Data science combines statistics, programming, and domain expertise.",
            "The Internet of Things connects billions of devices globally.",
            "Quantum computing promises exponential speedup for certain problems.",
            "Blockchain technology provides secure and transparent record keeping.",
            "Cloud computing offers scalable on-demand computing resources."
        ]

        self.synonyms = {
            'learn': ['understand', 'comprehend', 'grasp', 'acquire knowledge'],
            'enable': ['allow', 'permit', 'facilitate', 'empower'],
            'transform': ['change', 'revolutionize', 'reshape', 'modify'],
            'use': ['utilize', 'employ', 'apply', 'leverage'],
            'analyze': ['examine', 'study', 'investigate', 'evaluate'],
            'understand': ['comprehend', 'interpret', 'process', 'decipher'],
            'connect': ['link', 'join', 'attach', 'integrate'],
            'provide': ['offer', 'supply', 'deliver', 'furnish'],
            'promise': ['offer', 'provide', 'ensure', 'guarantee'],
            'offer': ['provide', 'supply', 'present', 'furnish']
        }

    def generate_original_texts(self, num_samples: int = 100) -> List[str]:
        texts = []
        for _ in range(num_samples):
            num_parts = random.randint(1, 3)
            base = random.sample(self.base_texts, num_parts)
            text = ' '.join(base)

            if random.random() > 0.5:
                text += " This is an important concept in modern technology."
            if random.random() > 0.7:
                text += " Many experts study this field extensively."

            texts.append(text)
        return texts

    def generate_plagiarized_texts(self, original_texts: List[str],
                                    num_samples: int = 100) -> List[str]:
        plagiarized = []

        for _ in range(num_samples):
            original = random.choice(original_texts)
            technique = random.choice(['synonym', 'restructure', 'paraphrase'])

            if technique == 'synonym':
                words = original.split()
                for i, word in enumerate(words):
                    if word.lower() in self.synonyms and random.random() > 0.5:
                        words[i] = random.choice(self.synonyms[word.lower()])
                plagiarized.append(' '.join(words))

            elif technique == 'restructure':
                sentences = original.split('. ')
                if len(sentences) > 1:
                    random.shuffle(sentences)
                plagiarized.append('. '.join(sentences))

            else:  # paraphrase
                words = original.split()
                if len(words) > 5:
                    if random.random() > 0.5:
                        remove_idx = random.randint(0, len(words) - 1)
                        words.pop(remove_idx)
                    if random.random() > 0.5:
                        words.insert(random.randint(0, len(words)), 'very')
                    if random.random() > 0.7:
                        words.insert(random.randint(0, len(words)), 'extremely')
                plagiarized.append(' '.join(words))

        return plagiarized

    def create_dataset(self, num_original: int = 100,
                        num_plagiarized: int = 100) -> Tuple[List[str], List[int]]:
        """Create a complete, shuffled, balanced dataset."""
        original_texts = self.generate_original_texts(num_original)
        plagiarized_texts = self.generate_plagiarized_texts(original_texts, num_plagiarized)

        texts = original_texts + plagiarized_texts
        labels = [0] * len(original_texts) + [1] * len(plagiarized_texts)

        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)

        return list(texts), list(labels)

    def save_dataset(self, texts: List[str], labels: List[int],
                      filepath: str = "data/plagiarism_dataset.csv") -> pd.DataFrame:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        df = pd.DataFrame({'text': texts, 'label': labels})
        df.to_csv(filepath, index=False)
        print(f"✅ Dataset saved to {filepath}")
        return df

    def load_dataset(self, filepath: str) -> Tuple[List[str], List[int]]:
        df = pd.read_csv(filepath)
        return df['text'].tolist(), df['label'].tolist()


if __name__ == "__main__":
    print("📊 Generating plagiarism dataset...")

    generator = PlagiarismDataGenerator()
    texts, labels = generator.create_dataset(num_original=50, num_plagiarized=50)

    print(f"   Total samples: {len(texts)}")
    print(f"   Original: {labels.count(0)}")
    print(f"   Plagiarized: {labels.count(1)}")

    generator.save_dataset(texts, labels)

    print("\n📊 Sample data:")
    for i in range(5):
        print(f"\nSample {i + 1}:")
        print(f"Text: {texts[i][:100]}...")
        print(f"Label: {'Plagiarized' if labels[i] == 1 else 'Original'}")
