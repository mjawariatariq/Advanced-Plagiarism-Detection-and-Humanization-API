




"""
Text Humanizer using the Google Gemini API.

Note: this was previously named `GrokHumanizer` even though it only ever
implemented Gemini support - there was no Grok/OpenAI integration anywhere
in the codebase, yet `api_integration.py` tried to call a nonexistent
`humanize_with_openai()` method on it, which would crash at runtime.
The class is renamed to `GeminiHumanizer` to match what it actually does,
with `GrokHumanizer` kept as an alias so any external references don't break.

FIX (this version): model name updated from "gemini-2.5-flash" to
"gemini-flash-latest". Google blocked gemini-2.5-flash for new API keys
ahead of its official Oct 16, 2026 shutdown, causing every humanize() call
to fail with a 404. "gemini-flash-latest" is an alias Google maintains that
always points at the current recommended flash model, so it won't need to
be manually bumped again on the next model cutover.
"""

import os
import re
import time
from typing import Dict, Optional
from datetime import datetime

from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(dotenv_path=".env", encoding="utf-8")


class GeminiHumanizer:
    """
    Text humanization backed by Google Gemini.
    Uses gemini-flash-latest exclusively - no fallback to another model.
    If the model is unreachable, `self.available` is False and
    humanize() returns a clear error instead of silently using a different model.
    """

    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = None
        self.gemini_model_name = "gemini-flash-latest"

        self.available = self._check_gemini_available()

        if self.available:
            self._init_gemini()
        else:
            print("⚠️ Gemini API key not found or invalid")
            print("💡 Please set GEMINI_API_KEY in .env file")

        self.styles = {
            'academic': 'more academic and formal',
            'casual': 'more casual and conversational',
            'professional': 'more professional and business-like',
            'creative': 'more creative and engaging',
            'simple': 'simpler and more readable'
        }

        self.humanization_history = []
        self.api_performance = {}

    def _check_gemini_available(self) -> bool:
        return bool(self.gemini_api_key) and self.gemini_api_key not in ("", "your-gemini-api-key-here") \
            and len(self.gemini_api_key) > 10

    def _init_gemini(self):
        """
        Configure the Gemini client. Availability is decided purely by
        whether a well-formed API key is present (checked in
        _check_gemini_available). We deliberately do NOT run a live
        "test" API call here to decide availability - a transient
        network blip, rate limit, or momentary API hiccup at startup
        used to permanently flip `self.available` to False even with a
        perfectly valid key, making Gemini look "Not Available" for the
        rest of the session for no real reason. Any genuine problem with
        the key/model still surfaces immediately and clearly the moment
        humanize() is actually called, via the try/except there.
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai
            print(f"   ✅ Gemini client configured: {self.gemini_model_name}")
        except ImportError:
            print("   ❌ google-generativeai not installed")
            print("   💡 Install with: pip install google-generativeai")
            self.available = False
        except Exception as e:
            print(f"   ❌ Gemini initialization error: {str(e)[:80]}")
            self.available = False

    def humanize(self, text: str, style: str = "academic",
                 similarity_calculator=None) -> Dict:
        """Humanize text using Gemini."""
        if not self.available:
            return {"success": False, "error": "Gemini API not available. Please set GEMINI_API_KEY in .env file"}

        if not text or len(text) < 5:
            return {"success": False, "error": "Text too short (minimum 5 characters)"}

        if style not in self.styles:
            style = "academic"

        try:
            start_time = time.time()
            style_description = self.styles.get(style, 'natural and human-like')

            # CRITICAL FIX: Force Gemini to return ONLY the rewritten text
            prompt = f"""Rewrite the following text to make it more {style_description}.

CRITICAL INSTRUCTIONS:
1. Return ONLY the rewritten text
2. Do NOT include any headers, labels, or explanations
3. Do NOT use markdown formatting
4. Do NOT include phrases like "Here is" or "Rewritten version"
5. Just return the rewritten text itself, nothing else

Text to rewrite:
{text}

Rewritten text:"""

            model = self.gemini_client.GenerativeModel(self.gemini_model_name)
            # Scale token budget to input length
            max_tokens = min(1024, max(400, len(text.split()) * 6))
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": max_tokens, "top_p": 0.95}
            )

            response_time = time.time() - start_time

            # Detect truncated response
            finish_reason = None
            try:
                if response.candidates:
                    raw_reason = getattr(response.candidates[0], 'finish_reason', None)
                    finish_reason = getattr(raw_reason, 'name', raw_reason)
            except Exception:
                pass

            if finish_reason is not None and str(finish_reason).upper() in ('MAX_TOKENS', '2'):
                return {
                    "success": False,
                    "error": "Gemini's response was cut off (hit the output token limit) before "
                             "finishing the rewrite. Try shorter input text or try again."
                }

            if not (response and response.text):
                return {"success": False, "error": "No response from Gemini"}

            humanized_text = self._clean_text(response.text)

            # If cleaning removed everything, try to extract the actual content
            if not humanized_text or len(humanized_text) < 5:
                # Try to find the actual rewritten content
                raw_text = response.text
                # Look for content after common headers
                headers = [
                    "rewritten text:",
                    "rewritten version:",
                    "humanized text:",
                    "humanized version:",
                    "here is",
                    "here are",
                    "output:",
                    "result:"
                ]
                for header in headers:
                    if header.lower() in raw_text.lower():
                        parts = raw_text.lower().split(header.lower(), 1)
                        if len(parts) > 1:
                            humanized_text = parts[1].strip()
                            break
                
                # If still empty, use the original text
                if not humanized_text or len(humanized_text) < 5:
                    humanized_text = text
                    print("⚠️ Could not extract rewritten text, using original")

            result = {
                "success": True,
                "original_text": text,
                "humanized_text": humanized_text,
                "api_used": "Google Gemini",
                "model_used": self.gemini_model_name,
                "response_time": response_time,
                "word_count": len(humanized_text.split()),
                "style": style,
                "timestamp": datetime.now().isoformat()
            }

            if similarity_calculator:
                try:
                    similarity = similarity_calculator(text, humanized_text)
                    result["similarity_to_original"] = float(similarity)
                    result["similarity_recalculated"] = True

                    self_sim = similarity_calculator(text, text)
                    result["original_self_similarity"] = float(self_sim)
                    result["similarity_change"] = float(similarity - self_sim)
                    result["humanization_effective"] = similarity < 0.8
                except Exception as e:
                    print(f"   ⚠️ Similarity calculation failed: {e}")

            self.humanization_history.append(result)
            self._update_performance(result)

            return result

        except Exception as e:
            return {"success": False, "error": f"Gemini error: {str(e)}"}

    def humanize_and_compare(self, text: str, style: str = "academic",
                              similarity_calculator=None) -> Dict:
        """Humanize text and compare similarity before/after."""
        if not similarity_calculator:
            return {"success": False, "error": "Similarity calculator function required"}

        original_self_sim = similarity_calculator(text, text)
        result = self.humanize(text, style, similarity_calculator)

        if result.get('success', False):
            result['original_self_similarity'] = float(original_self_sim)
            result['similarity_after_humanization'] = result.get('similarity_to_original', 0)
            result['similarity_change'] = result['similarity_after_humanization'] - original_self_sim
            result['humanization_effective'] = result['similarity_after_humanization'] < 0.8

        return result

    def _clean_text(self, text: str) -> str:
    """Clean Gemini response - remove ALL headers and explanations."""
    text = text.strip()
    
    # Split by newlines and keep only content that looks like actual text
    lines = text.split('\n')
    result_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip ANY line that looks like a header/explanation
        line_lower = line.lower()
        skip_patterns = [
            "here are a few",
            "here's a rewritten",
            "rewritten version",
            "humanized version",
            "here is the rewritten",
            "rewritten text",
            "output:",
            "result:",
            "**rewritten",
            "**humanized",
            "**here",
            "depending on",
            "desired level",
            "academic formality",
            "tone you want",
            "style you prefer"
        ]
        
        should_skip = False
        for pattern in skip_patterns:
            if pattern in line_lower:
                should_skip = True
                break
        
        if not should_skip:
            result_lines.append(line)
    
    # If we removed everything, return the original
    if not result_lines:
        return text
    
    result = ' '.join(result_lines)
    
    # Remove markdown and quotes
    result = re.sub(r'\*\*', '', result)
    result = re.sub(r'#+\s*', '', result)
    
    if (result.startswith('"') and result.endswith('"')) or (result.startswith("'") and result.endswith("'")):
        result = result[1:-1]
    
    return result.strip()

    def _update_performance(self, result: Dict):
        response_time = result.get('response_time', 0)
        stats = self.api_performance.setdefault('Gemini', {
            'total_requests': 0, 'total_time': 0, 'avg_response_time': 0
        })
        stats['total_requests'] += 1
        stats['total_time'] += response_time
        stats['avg_response_time'] = stats['total_time'] / stats['total_requests']

    def get_statistics(self) -> Dict:
        total = len(self.humanization_history)
        if total == 0:
            return {
                "total_humanizations": 0, "success_rate": 0, "avg_response_time": 0,
                "api_usage": {}, "styles_used": []
            }

        success_count = sum(1 for h in self.humanization_history if h.get('success', False))
        avg_time = sum(h.get('response_time', 0) for h in self.humanization_history) / total

        api_usage = {}
        for h in self.humanization_history:
            api = h.get('api_used', 'unknown')
            api_usage[api] = api_usage.get(api, 0) + 1

        return {
            "total_humanizations": total,
            "success_rate": success_count / total * 100,
            "avg_response_time": avg_time,
            "api_usage": api_usage,
            "styles_used": list(set(h.get('style', 'unknown') for h in self.humanization_history)),
            "api_performance": self.api_performance
        }

    def get_api_status(self) -> Dict:
        return {
            'gemini': {
                'available': self.available,
                'model': self.gemini_model_name,
                'api_key_set': bool(self.gemini_api_key)
            }
        }

    def humanize_with_gemini(self, text: str, style: str = "academic") -> Dict:
        """Alias for humanize()."""
        return self.humanize(text, style)

    def set_api_key(self, api_key: str) -> None:
        self.gemini_api_key = api_key
        self.available = self._check_gemini_available()
        if self.available:
            self._init_gemini()
            print("✅ Gemini API key updated successfully")
        else:
            print("⚠️ Invalid API key provided")


# Backward-compatible alias - old code/imports referencing GrokHumanizer keep working.
GrokHumanizer = GeminiHumanizer
