# """
# FastAPI service for plagiarism detection and AI-text humanization.

# Fixes vs. the original code:
#  - `/humanize` and `/full-pipeline` branched on `input_data.api_choice` and
#    called `self.humanizer.humanize_with_openai(...)` for anything other than
#    "gemini" - but no such method, or any OpenAI integration, ever existed
#    anywhere in this codebase. That branch is removed; Gemini is the only
#    supported backend, matching what `text_humanizer.py` actually implements.
#  - `/detect` built `dataset_texts = self.detector.vectorizer.transform([]).shape[0] > 0`,
#    which is dead, meaningless code operating on a vectorizer attribute that
#    was never actually fitted, and its result was never used. Removed.
#  - `get_detailed_report(...)` returns a `PlagiarismReport` dataclass, which
#    FastAPI/Pydantic cannot serialize into `data: Optional[dict]` as-is - it
#    would raise a validation error on every `/detect` call. Now converted
#    via `.to_dict()`.
#  - `/batch` always appended '...' to preview text even when the text was
#    under 100 characters. Fixed to only truncate when needed.
# """

# from datetime import datetime
# from typing import List, Optional

# import uvicorn
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel


# class TextInput(BaseModel):
#     text: str
#     style: Optional[str] = "academic"


# class BatchTextInput(BaseModel):
#     texts: List[str]
#     style: Optional[str] = "academic"


# class CompareInput(BaseModel):
#     text1: str
#     text2: str


# class APIResponse(BaseModel):
#     status: str
#     message: str
#     data: Optional[dict] = None
#     timestamp: str


# class PlagiarismAPI:
#     """FastAPI-based API for plagiarism detection and humanization."""

#     # A tiny reference corpus used for the /detect endpoint demo. In
#     # production this should be swapped for a real document index.
#     SAMPLE_REFERENCE_TEXTS = [
#         "Artificial intelligence is the simulation of human intelligence in machines",
#         "Machine learning is a subset of artificial intelligence"
#     ]

#     def __init__(self, detector, humanizer):
#         self.detector = detector
#         self.humanizer = humanizer
#         self.app = FastAPI(title="Plagiarism Detection & Humanization API", version="1.0.0")
#         self.setup_routes()

#     def setup_routes(self):
#         @self.app.get("/")
#         def root():
#             return {
#                 "service": "Plagiarism Detection & AI Text Humanization API",
#                 "version": "1.0.0",
#                 "status": "running",
#                 "endpoints": ["/health", "/detect", "/humanize", "/full-pipeline", "/batch", "/compare"]
#             }

#         @self.app.get("/health")
#         def health():
#             return {
#                 "status": "healthy",
#                 "timestamp": datetime.now().isoformat(),
#                 "models_loaded": self.detector.is_loaded,
#                 "gemini_available": self.humanizer.available
#             }

#         @self.app.post("/detect", response_model=APIResponse)
#         def detect_plagiarism(input_data: TextInput):
#             try:
#                 prediction = self.detector.predict(input_data.text)
#                 report = self.detector.get_detailed_report(input_data.text, self.SAMPLE_REFERENCE_TEXTS)

#                 return APIResponse(
#                     status="success",
#                     message="Plagiarism detection completed",
#                     data={
#                         "prediction": prediction,
#                         "report": report.to_dict(),
#                         "text": input_data.text
#                     },
#                     timestamp=datetime.now().isoformat()
#                 )
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=str(e))

#         @self.app.post("/humanize", response_model=APIResponse)
#         def humanize_text(input_data: TextInput):
#             try:
#                 result = self.humanizer.humanize(
#                     input_data.text, input_data.style,
#                     similarity_calculator=self.detector.calculate_semantic_similarity
#                 )

#                 if not result.get('success', False):
#                     raise HTTPException(status_code=400, detail=result.get('error', 'Humanization failed'))

#                 return APIResponse(
#                     status="success",
#                     message="Text humanization completed",
#                     data=result,
#                     timestamp=datetime.now().isoformat()
#                 )
#             except HTTPException:
#                 raise
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=str(e))

#         @self.app.post("/full-pipeline", response_model=APIResponse)
#         def full_pipeline(input_data: TextInput):
#             try:
#                 prediction = self.detector.predict(input_data.text)

#                 # Gemini always runs here, on every text, regardless of
#                 # whether plagiarism was detected.
#                 humanized = None
#                 result = self.humanizer.humanize(
#                     input_data.text, input_data.style,
#                     similarity_calculator=self.detector.calculate_semantic_similarity
#                 )
#                 if result.get('success', False):
#                     humanized = result

#                 return APIResponse(
#                     status="success",
#                     message="Full pipeline completed",
#                     data={
#                         "original_text": input_data.text,
#                         "plagiarism_check": prediction,
#                         "humanized_text": humanized,
#                         "style": input_data.style
#                     },
#                     timestamp=datetime.now().isoformat()
#                 )
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=str(e))

#         @self.app.post("/compare", response_model=APIResponse)
#         def compare_texts(input_data: CompareInput):
#             try:
#                 report = self.detector.compare_texts(input_data.text1, input_data.text2)
#                 return APIResponse(
#                     status="success",
#                     message="Comparison completed",
#                     data=report.to_dict(),
#                     timestamp=datetime.now().isoformat()
#                 )
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=str(e))

#         @self.app.post("/batch", response_model=APIResponse)
#         def batch_process(input_data: BatchTextInput):
#             try:
#                 results = []
#                 for text in input_data.texts:
#                     prediction = self.detector.predict(text)
#                     preview = text[:100] + '...' if len(text) > 100 else text
#                     results.append({'text': preview, 'prediction': prediction})

#                 return APIResponse(
#                     status="success",
#                     message=f"Batch processed {len(results)} texts",
#                     data={'total_processed': len(results), 'results': results, 'style': input_data.style},
#                     timestamp=datetime.now().isoformat()
#                 )
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=str(e))

#     def run(self, host="0.0.0.0", port=8000):
#         uvicorn.run(self.app, host=host, port=port)


# def create_app() -> FastAPI:
#     """Factory used by `uvicorn api_integration:app` / production servers.

#     Loads the pretrained jpwahle/longformer-base-plagiarism-detection model
#     directly - no training or dataset generation required. A fine-tuned
#     model at ./models/fine_tuned_model is used automatically if present,
#     but is entirely optional.
#     """
#     from src.huggingface_detector import HuggingFaceDetector
#     from src.text_humanizer import GeminiHumanizer
#     import os

#     detector = HuggingFaceDetector()
#     fine_tuned_path = "./models/fine_tuned_model"
#     if os.path.exists(fine_tuned_path):
#         detector.load_fine_tuned_model(fine_tuned_path)
#     else:
#         detector.load_model()

#     humanizer = GeminiHumanizer()
#     api = PlagiarismAPI(detector, humanizer)
#     return api.app


# app = None
# try:
#     app = create_app()
# except Exception as _e:  # pragma: no cover - only triggers without deps/model access
#     print(f"⚠️ Could not eagerly initialize API app: {_e}")
#     print("   The app will need to be created manually via create_app().")


# if __name__ == "__main__":
#     if app is None:
#         app = create_app()
#     uvicorn.run(app, host="0.0.0.0", port=8000)



"""
FastAPI service for plagiarism detection and AI-text humanization.

Fixes vs. the original code:
 - `/humanize` and `/full-pipeline` branched on `input_data.api_choice` and
   called `self.humanizer.humanize_with_openai(...)` for anything other than
   "gemini" - but no such method, or any OpenAI integration, ever existed
   anywhere in this codebase. That branch is removed; Gemini is the only
   supported backend, matching what `text_humanizer.py` actually implements.
 - `/detect` built `dataset_texts = self.detector.vectorizer.transform([]).shape[0] > 0`,
   which is dead, meaningless code operating on a vectorizer attribute that
   was never actually fitted, and its result was never used. Removed.
 - `get_detailed_report(...)` returns a `PlagiarismReport` dataclass, which
   FastAPI/Pydantic cannot serialize into `data: Optional[dict]` as-is - it
   would raise a validation error on every `/detect` call. Now converted
   via `.to_dict()`.
 - `/batch` always appended '...' to preview text even when the text was
   under 100 characters. Fixed to only truncate when needed.
"""

import os
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.reference_corpus import load_reference_corpus


class TextInput(BaseModel):
    text: str
    style: Optional[str] = "academic"


class BatchTextInput(BaseModel):
    texts: List[str]
    style: Optional[str] = "academic"


class CompareInput(BaseModel):
    text1: str
    text2: str


class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None
    timestamp: str


class PlagiarismAPI:
    """FastAPI-based API for plagiarism detection and humanization."""

    def __init__(self, detector, humanizer):
        self.detector = detector
        self.humanizer = humanizer
        # Loaded from data/reference_corpus.txt if present, else a tiny
        # built-in placeholder (see src/reference_corpus.py). This is
        # what "Plagiarized" verdicts are grounded against in /detect
        # and /full-pipeline - drop real source documents into that file
        # for meaningful results.
        self.reference_texts = load_reference_corpus()
        self.app = FastAPI(title="Plagiarism Detection & Humanization API", version="1.0.0")
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        def root():
            return {
                "service": "Plagiarism Detection & AI Text Humanization API",
                "version": "1.0.0",
                "status": "running",
                "endpoints": ["/health", "/detect", "/humanize", "/full-pipeline", "/batch", "/compare"]
            }

        @self.app.get("/health")
        def health():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": self.detector.is_loaded,
                "gemini_available": self.humanizer.available
            }

        @self.app.post("/detect", response_model=APIResponse)
        def detect_plagiarism(input_data: TextInput):
            try:
                # predict_grounded() only lets a "Plagiarized" verdict
                # stand if there's an actual matching document in the
                # reference corpus - otherwise it's downgraded to
                # "Style-Flagged (No Matching Source)" instead of a hard
                # Plagiarized/Critical Risk result.
                prediction = self.detector.predict_grounded(input_data.text, self.reference_texts)
                report = self.detector.get_detailed_report(input_data.text, self.reference_texts)

                return APIResponse(
                    status="success",
                    message="Plagiarism detection completed",
                    data={
                        "prediction": prediction,
                        "report": report.to_dict(),
                        "text": input_data.text
                    },
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/humanize", response_model=APIResponse)
        def humanize_text(input_data: TextInput):
            try:
                result = self.humanizer.humanize(
                    input_data.text, input_data.style,
                    similarity_calculator=self.detector.calculate_semantic_similarity
                )

                if not result.get('success', False):
                    raise HTTPException(status_code=400, detail=result.get('error', 'Humanization failed'))

                return APIResponse(
                    status="success",
                    message="Text humanization completed",
                    data=result,
                    timestamp=datetime.now().isoformat()
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/full-pipeline", response_model=APIResponse)
        def full_pipeline(input_data: TextInput):
            try:
                prediction = self.detector.predict_grounded(input_data.text, self.reference_texts)

                # Gemini always runs here, on every text, regardless of
                # whether plagiarism was detected.
                humanized = None
                result = self.humanizer.humanize(
                    input_data.text, input_data.style,
                    similarity_calculator=self.detector.calculate_semantic_similarity
                )
                if result.get('success', False):
                    humanized = result

                return APIResponse(
                    status="success",
                    message="Full pipeline completed",
                    data={
                        "original_text": input_data.text,
                        "plagiarism_check": prediction,
                        "humanized_text": humanized,
                        "style": input_data.style
                    },
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/compare", response_model=APIResponse)
        def compare_texts(input_data: CompareInput):
            try:
                report = self.detector.compare_texts(input_data.text1, input_data.text2)
                return APIResponse(
                    status="success",
                    message="Comparison completed",
                    data=report.to_dict(),
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch", response_model=APIResponse)
        def batch_process(input_data: BatchTextInput):
            try:
                results = []
                for text in input_data.texts:
                    prediction = self.detector.predict(text)
                    preview = text[:100] + '...' if len(text) > 100 else text
                    results.append({'text': preview, 'prediction': prediction})

                return APIResponse(
                    status="success",
                    message=f"Batch processed {len(results)} texts",
                    data={'total_processed': len(results), 'results': results, 'style': input_data.style},
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)


def create_app() -> FastAPI:
    """Factory used by `uvicorn api_integration:app` / production servers.

    Loads the pretrained jpwahle/longformer-base-plagiarism-detection model
    directly - no training or dataset generation required. A fine-tuned
    model at ./models/fine_tuned_model is used automatically if present,
    but is entirely optional.
    """
    from src.huggingface_detector import HuggingFaceDetector
    from src.text_humanizer import GeminiHumanizer
    import os

    detector = HuggingFaceDetector()
    fine_tuned_path = "./models/fine_tuned_model"
    if os.path.exists(fine_tuned_path):
        detector.load_fine_tuned_model(fine_tuned_path)
    else:
        detector.load_model()

    humanizer = GeminiHumanizer()
    api = PlagiarismAPI(detector, humanizer)
    return api.app


app = None
try:
    app = create_app()
except Exception as _e:  # pragma: no cover - only triggers without deps/model access
    print(f"⚠️ Could not eagerly initialize API app: {_e}")
    print("   The app will need to be created manually via create_app().")


if __name__ == "__main__":
    if app is None:
        app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)