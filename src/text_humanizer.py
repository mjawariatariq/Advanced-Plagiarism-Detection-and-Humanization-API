


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

            prompt = f"""Rewrite the following text to make it more {style_description}.
Improve grammar, sentence structure, and readability while preserving the original meaning.

Important: Ensure the rewritten text is original and not a direct copy. Make it sound like a human wrote it.

Text to rewrite:
{text}

Rewritten version:"""

            model = self.gemini_client.GenerativeModel(self.gemini_model_name)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": 500, "top_p": 0.95}
            )

            response_time = time.time() - start_time

            if not (response and response.text):
                return {"success": False, "error": "No response from Gemini"}

            humanized_text = self._clean_text(response.text)

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
        text = text.strip()
        prefixes = [
            r'^Rewritten\s+version:?\s*',
            r"^Here'?s\s+the\s+rewritten\s+text:?\s*",
            r'^Humanized\s+version:?\s*',
            r'^Revised\s+text:?\s*',
            r'^Output:?\s*',
            r'^Result:?\s*',
            r'^Response:?\s*'
        ]
        for prefix in prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)

        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]

        return text.strip()

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









# """
# Text Humanizer using the Google Gemini API.

# Note: this was previously named `GrokHumanizer` even though it only ever
# implemented Gemini support - there was no Grok/OpenAI integration anywhere
# in the codebase, yet `api_integration.py` tried to call a nonexistent
# `humanize_with_openai()` method on it, which would crash at runtime.
# The class is renamed to `GeminiHumanizer` to match what it actually does,
# with `GrokHumanizer` kept as an alias so any external references don't break.

# FIX (this version): model name updated from "gemini-2.5-flash" to
# "gemini-flash-latest". Google blocked gemini-2.5-flash for new API keys
# ahead of its official Oct 16, 2026 shutdown, causing every humanize() call
# to fail with a 404. "gemini-flash-latest" is an alias Google maintains that
# always points at the current recommended flash model, so it won't need to
# be manually bumped again on the next model cutover.
# """

# import os
# import re
# import time
# from typing import Dict, Optional
# from datetime import datetime

# from dotenv import load_dotenv

# if os.path.exists(".env"):
#     load_dotenv(dotenv_path=".env", encoding="utf-8")


# class GeminiHumanizer:
#     """
#     Text humanization backed by Google Gemini.
#     Uses gemini-flash-latest exclusively - no fallback to another model.
#     If the model is unreachable, `self.available` is False and
#     humanize() returns a clear error instead of silently using a different model.
#     """

#     def __init__(self):
#         self.gemini_api_key = os.getenv("GEMINI_API_KEY")
#         self.gemini_client = None
#         self.gemini_model_name = "gemini-flash-latest"

#         self.available = self._check_gemini_available()

#         if self.available:
#             self._init_gemini()
#         else:
#             print("⚠️ Gemini API key not found or invalid")
#             print("💡 Please set GEMINI_API_KEY in .env file")

#         self.styles = {
#             'academic': 'more academic and formal',
#             'casual': 'more casual and conversational',
#             'professional': 'more professional and business-like',
#             'creative': 'more creative and engaging',
#             'simple': 'simpler and more readable'
#         }

#         self.humanization_history = []
#         self.api_performance = {}

#     def _check_gemini_available(self) -> bool:
#         return bool(self.gemini_api_key) and self.gemini_api_key not in ("", "your-gemini-api-key-here") \
#             and len(self.gemini_api_key) > 10

#     def _init_gemini(self):
#         """
#         Configure the Gemini client. Availability is decided purely by
#         whether a well-formed API key is present (checked in
#         _check_gemini_available). We deliberately do NOT run a live
#         "test" API call here to decide availability - a transient
#         network blip, rate limit, or momentary API hiccup at startup
#         used to permanently flip `self.available` to False even with a
#         perfectly valid key, making Gemini look "Not Available" for the
#         rest of the session for no real reason. Any genuine problem with
#         the key/model still surfaces immediately and clearly the moment
#         humanize() is actually called, via the try/except there.
#         """
#         try:
#             import google.generativeai as genai
#             genai.configure(api_key=self.gemini_api_key)
#             self.gemini_client = genai
#             print(f"   ✅ Gemini client configured: {self.gemini_model_name}")
#         except ImportError:
#             print("   ❌ google-generativeai not installed")
#             print("   💡 Install with: pip install google-generativeai")
#             self.available = False
#         except Exception as e:
#             print(f"   ❌ Gemini initialization error: {str(e)[:80]}")
#             self.available = False

#     def humanize(self, text: str, style: str = "academic",
#                  similarity_calculator=None) -> Dict:
#         """Humanize text using Gemini."""
#         if not self.available:
#             return {"success": False, "error": "Gemini API not available. Please set GEMINI_API_KEY in .env file"}

#         if not text or len(text) < 5:
#             return {"success": False, "error": "Text too short (minimum 5 characters)"}

#         if style not in self.styles:
#             style = "academic"

#         try:
#             start_time = time.time()
#             style_description = self.styles.get(style, 'natural and human-like')

#             # Explicitly forbid headers/markdown/preamble in the prompt.
#             # Previously Gemini would sometimes prepend a markdown header
#             # like "**Rewritten Version (Academic structure):**" before
#             # the actual rewrite. `_clean_text()`'s old prefix-regexes
#             # only matched plain text like "Rewritten version:" and never
#             # matched that markdown-wrapped form, so the header (and its
#             # wasted tokens) survived into the final output, and on top
#             # of that could push the real rewrite past max_output_tokens
#             # and cut it off mid-sentence.
#             prompt = f"""Rewrite the following text to make it more {style_description}.
# Improve grammar, sentence structure, and readability while preserving the original meaning.

# Important: Ensure the rewritten text is original and not a direct copy. Make it sound like a human wrote it.

# Respond with ONLY the rewritten text itself. Do not include a heading, label, explanation, markdown formatting (no ** or #), or surrounding quotation marks.

# Text to rewrite:
# {text}

# Rewritten text:"""

#             model = self.gemini_client.GenerativeModel(self.gemini_model_name)
#             # Scale the token budget to the input length instead of a flat
#             # 500. A flat cap was enough to silently truncate the response
#             # mid-sentence for longer inputs (or when Gemini added a
#             # header eating into the budget), producing garbled output
#             # with no indication anything was cut off.
#             max_tokens = min(1024, max(400, len(text.split()) * 6))
#             response = model.generate_content(
#                 prompt,
#                 generation_config={"temperature": 0.7, "max_output_tokens": max_tokens, "top_p": 0.95}
#             )

#             response_time = time.time() - start_time

#             # Detect a truncated response instead of silently returning a
#             # cut-off, garbled rewrite. finish_reason == 2 / "MAX_TOKENS"
#             # means Gemini ran out of output budget mid-generation.
#             finish_reason = None
#             try:
#                 if response.candidates:
#                     raw_reason = getattr(response.candidates[0], 'finish_reason', None)
#                     finish_reason = getattr(raw_reason, 'name', raw_reason)
#             except Exception:
#                 pass

#             if finish_reason is not None and str(finish_reason).upper() in ('MAX_TOKENS', '2'):
#                 return {
#                     "success": False,
#                     "error": "Gemini's response was cut off (hit the output token limit) before "
#                              "finishing the rewrite. Try shorter input text or try again."
#                 }

#             if not (response and response.text):
#                 return {"success": False, "error": "No response from Gemini"}

#             humanized_text = self._clean_text(response.text)

#             result = {
#                 "success": True,
#                 "original_text": text,
#                 "humanized_text": humanized_text,
#                 "api_used": "Google Gemini",
#                 "model_used": self.gemini_model_name,
#                 "response_time": response_time,
#                 "word_count": len(humanized_text.split()),
#                 "style": style,
#                 "timestamp": datetime.now().isoformat()
#             }

#             if similarity_calculator:
#                 try:
#                     similarity = similarity_calculator(text, humanized_text)
#                     result["similarity_to_original"] = float(similarity)
#                     result["similarity_recalculated"] = True

#                     self_sim = similarity_calculator(text, text)
#                     result["original_self_similarity"] = float(self_sim)
#                     result["similarity_change"] = float(similarity - self_sim)
#                     result["humanization_effective"] = similarity < 0.8
#                 except Exception as e:
#                     print(f"   ⚠️ Similarity calculation failed: {e}")

#             self.humanization_history.append(result)
#             self._update_performance(result)

#             return result

#         except Exception as e:
#             return {"success": False, "error": f"Gemini error: {str(e)}"}

#     def humanize_and_compare(self, text: str, style: str = "academic",
#                               similarity_calculator=None) -> Dict:
#         """Humanize text and compare similarity before/after."""
#         if not similarity_calculator:
#             return {"success": False, "error": "Similarity calculator function required"}

#         original_self_sim = similarity_calculator(text, text)
#         result = self.humanize(text, style, similarity_calculator)

#         if result.get('success', False):
#             result['original_self_similarity'] = float(original_self_sim)
#             result['similarity_after_humanization'] = result.get('similarity_to_original', 0)
#             result['similarity_change'] = result['similarity_after_humanization'] - original_self_sim
#             result['humanization_effective'] = result['similarity_after_humanization'] < 0.8

#         return result

#     def _clean_text(self, text: str) -> str:
#         text = text.strip()

#         # Drop leading meta/header lines Gemini sometimes prepends, e.g.
#         # "**Rewritten Version (Academic structure):**". The old
#         # prefix-regexes below only matched plain text like "Rewritten
#         # version:" starting at position 0 - they never matched a
#         # markdown-bolded header, so it (and the tokens it cost) leaked
#         # into the final output.
#         lines = text.split('\n')
#         while lines:
#             first = lines[0].strip()
#             looks_like_header = (
#                 not first
#                 or (first.startswith('**') and first.endswith('**') and len(first) < 80)
#                 or first.startswith('#')
#                 or bool(re.match(r'^\*+\s*\(?[A-Za-z ]{0,40}\)?:?\s*\**$', first))
#                 or (first.endswith(':') and len(first) < 60 and not first.startswith('"'))
#             )
#             if looks_like_header:
#                 lines.pop(0)
#             else:
#                 break
#         text = '\n'.join(lines).strip()

#         prefixes = [
#             r'^Rewritten\s+version:?\s*',
#             r"^Here'?s\s+the\s+rewritten\s+text:?\s*",
#             r'^Humanized\s+version:?\s*',
#             r'^Revised\s+text:?\s*',
#             r'^Output:?\s*',
#             r'^Result:?\s*',
#             r'^Response:?\s*'
#         ]
#         for prefix in prefixes:
#             text = re.sub(prefix, '', text, flags=re.IGNORECASE)
#         text = text.strip()

#         # Strip markdown bold/italic wrapping the *entire* remaining
#         # response (e.g. "**Despite having switched...**").
#         while len(text) > 4 and (
#             (text.startswith('**') and text.endswith('**')) or
#             (text.startswith('*') and text.endswith('*'))
#         ):
#             wrap_len = 2 if text.startswith('**') else 1
#             text = text[wrap_len:-wrap_len].strip()

#         if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
#             text = text[1:-1]

#         return text.strip()

#     def _update_performance(self, result: Dict):
#         response_time = result.get('response_time', 0)
#         stats = self.api_performance.setdefault('Gemini', {
#             'total_requests': 0, 'total_time': 0, 'avg_response_time': 0
#         })
#         stats['total_requests'] += 1
#         stats['total_time'] += response_time
#         stats['avg_response_time'] = stats['total_time'] / stats['total_requests']

#     def get_statistics(self) -> Dict:
#         total = len(self.humanization_history)
#         if total == 0:
#             return {
#                 "total_humanizations": 0, "success_rate": 0, "avg_response_time": 0,
#                 "api_usage": {}, "styles_used": []
#             }

#         success_count = sum(1 for h in self.humanization_history if h.get('success', False))
#         avg_time = sum(h.get('response_time', 0) for h in self.humanization_history) / total

#         api_usage = {}
#         for h in self.humanization_history:
#             api = h.get('api_used', 'unknown')
#             api_usage[api] = api_usage.get(api, 0) + 1

#         return {
#             "total_humanizations": total,
#             "success_rate": success_count / total * 100,
#             "avg_response_time": avg_time,
#             "api_usage": api_usage,
#             "styles_used": list(set(h.get('style', 'unknown') for h in self.humanization_history)),
#             "api_performance": self.api_performance
#         }

#     def get_api_status(self) -> Dict:
#         return {
#             'gemini': {
#                 'available': self.available,
#                 'model': self.gemini_model_name,
#                 'api_key_set': bool(self.gemini_api_key)
#             }
#         }

#     def humanize_with_gemini(self, text: str, style: str = "academic") -> Dict:
#         """Alias for humanize()."""
#         return self.humanize(text, style)

#     def set_api_key(self, api_key: str) -> None:
#         self.gemini_api_key = api_key
#         self.available = self._check_gemini_available()
#         if self.available:
#             self._init_gemini()
#             print("✅ Gemini API key updated successfully")
#         else:
#             print("⚠️ Invalid API key provided")


# # Backward-compatible alias - old code/imports referencing GrokHumanizer keep working.
# GrokHumanizer = GeminiHumanizer