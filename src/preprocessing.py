"""
Text Preprocessing Module
"""

import string
from typing import Dict, List


class TextPreprocessor:
    """Text preprocessing for plagiarism detection"""

    def __init__(self):
        self.stop_words = self._get_stop_words()

    def _get_stop_words(self) -> set:
        """Get common English stop words"""
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'
        }

    def preprocess(self, text: str, remove_stopwords: bool = True,
                   lowercase: bool = True, remove_punctuation: bool = True) -> Dict:
        """Preprocess text with multiple options."""
        original = text
        processed = text

        if lowercase:
            processed = processed.lower()

        if remove_punctuation:
            processed = self._remove_punctuation(processed)

        if remove_stopwords:
            processed = self._remove_stopwords(processed)

        processed = self._normalize_whitespace(processed)

        return {
            'original_text': original,
            'cleaned_text': processed,
            'word_count': len(processed.split()),
            'original_word_count': len(original.split()),
            'preprocessing_applied': {
                'lowercase': lowercase,
                'remove_punctuation': remove_punctuation,
                'remove_stopwords': remove_stopwords
            }
        }

    def _remove_punctuation(self, text: str) -> str:
        return ''.join(char for char in text if char not in string.punctuation)

    def _remove_stopwords(self, text: str) -> str:
        words = text.split()
        filtered = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered)

    def _normalize_whitespace(self, text: str) -> str:
        return ' '.join(text.split())

    def clean_text(self, text: str) -> str:
        """Quick clean function"""
        return self._normalize_whitespace(self._remove_punctuation(text.lower()))

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords from text"""
        cleaned = self._remove_punctuation(text.lower())
        words = cleaned.split()
        from collections import Counter
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(top_n)
                if word not in self.stop_words]
