# # """
# # Streamlit UI - HUMA PLAG: Plagiarism Detection & AI Text Humanization

# # This is a UI layer only. All actual logic (model inference, grounding,
# # humanization, similarity) lives in the same src/ modules main.py (the
# # CLI) already uses - nothing is duplicated or reimplemented here.

# # Run with:
# #     streamlit run app.py
# # """

# # import os
# # import sys
# # import io
# # from datetime import datetime

# # import pandas as pd
# # import streamlit as st

# # # --- Path setup, mirrors main.py ---------------------------------------
# # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# # SRC_PATH = os.path.join(CURRENT_DIR, "src")
# # if SRC_PATH not in sys.path:
# #     sys.path.insert(0, SRC_PATH)

# # from src.preprocessing import TextPreprocessor
# # from src.huggingface_detector import HuggingFaceDetector
# # from src.text_humanizer import GeminiHumanizer
# # from src.reference_corpus import load_reference_corpus, DEFAULT_CORPUS_PATH

# # try:
# #     from dotenv import load_dotenv
# #     if os.path.exists(".env"):
# #         load_dotenv(dotenv_path=".env", encoding="utf-8")
# # except ImportError:
# #     pass


# # # ======================================================================
# # # Page config + styling
# # # ======================================================================
# # st.set_page_config(
# #     page_title="HUMA PLAG - Plagiarism Detection & Humanization",
# #     page_icon="📚",
# #     layout="wide",
# #     initial_sidebar_state="expanded",
# # )

# # st.markdown("""
# # <style>
# #     .block-container { padding-top: 2rem; padding-bottom: 3rem; }
# #     .risk-badge {
# #         display: inline-block; padding: 5px 16px; border-radius: 999px;
# #         color: white; font-weight: 600; font-size: 0.92rem;
# #     }
# #     .result-card {
# #         background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.18);
# #         border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
# #     }
# #     .status-pill {
# #         display: inline-block; padding: 3px 12px; border-radius: 999px;
# #         font-size: 0.82rem; font-weight: 600; margin-right: 6px;
# #     }
# #     .pill-ok { background: #16a34a22; color: #16a34a; }
# #     .pill-bad { background: #dc262622; color: #dc2626; }
# #     .ref-quote {
# #         border-left: 3px solid #6366f1; padding-left: 12px; margin: 8px 0;
# #         font-style: italic; opacity: 0.9;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # RISK_COLORS = {
# #     "Low Risk": "#16a34a",
# #     "Moderate Risk": "#ca8a04",
# #     "High Risk": "#ea580c",
# #     "Very High Risk": "#dc2626",
# #     "Critical Risk": "#991b1b",
# #     "Uncertain": "#6b7280",
# #     "Unconfirmed": "#a16207",
# #     "No Data": "#6b7280",
# #     "Error": "#6b7280",
# #     "Unknown": "#6b7280",
# # }


# # def risk_badge_html(risk_level: str) -> str:
# #     color = "#6b7280"
# #     for key, c in RISK_COLORS.items():
# #         if key in (risk_level or ""):
# #             color = c
# #             break
# #     return f'<span class="risk-badge" style="background:{color};">{risk_level}</span>'


# # # ======================================================================
# # # Cached resource loaders - models load once per server process
# # # ======================================================================
# # @st.cache_resource(show_spinner="🔄 Loading text preprocessor...")
# # def get_preprocessor():
# #     return TextPreprocessor()


# # @st.cache_resource(show_spinner="🔄 Loading plagiarism detection model (Longformer) - first load can take a minute...")
# # def get_detector():
# #     detector = HuggingFaceDetector()
# #     fine_tuned_path = os.path.join(CURRENT_DIR, "models", "fine_tuned_model")
# #     try:
# #         if os.path.exists(fine_tuned_path):
# #             detector.load_fine_tuned_model(fine_tuned_path)
# #         else:
# #             detector.load_model()
# #     except Exception as e:
# #         st.session_state["_detector_load_error"] = str(e)
# #     return detector


# # @st.cache_resource(show_spinner="🔄 Connecting to Gemini...")
# # def get_humanizer():
# #     return GeminiHumanizer()


# # preprocessor = get_preprocessor()
# # detector = get_detector()
# # humanizer = get_humanizer()

# # if "reference_texts" not in st.session_state:
# #     st.session_state.reference_texts = load_reference_corpus()
# # if "history" not in st.session_state:
# #     st.session_state.history = []


# # def log_history(entry_type: str, **kwargs):
# #     st.session_state.history.append({
# #         "type": entry_type,
# #         "timestamp": datetime.now().isoformat(),
# #         **kwargs
# #     })


# # # ======================================================================
# # # Shared render helpers
# # # ======================================================================
# # def render_detection_result(result: dict, semantic_sim: float = None):
# #     st.markdown('<div class="result-card">', unsafe_allow_html=True)
# #     c1, c2, c3 = st.columns(3)
# #     c1.metric("Status", result.get("label", "Unknown"))
# #     c2.metric("Confidence", f"{result.get('confidence', 0) * 100:.1f}%")
# #     with c3:
# #         st.markdown("**Risk Level**")
# #         st.markdown(risk_badge_html(result.get("risk_level", "Unknown")), unsafe_allow_html=True)

# #     if result.get("probability"):
# #         st.progress(
# #             min(max(result["probability"][1], 0.0), 1.0),
# #             text=f"Plagiarism probability: {result['probability'][1] * 100:.1f}%  "
# #                  f"(Original: {result['probability'][0] * 100:.1f}%)"
# #         )

# #     detail_cols = st.columns(4)
# #     detail_cols[0].caption(f"📊 Word Count: {result.get('word_count', 0)}")
# #     detail_cols[1].caption(f"🎯 Model: {result.get('model_type', 'Unknown')}")
# #     detail_cols[2].caption(f"📐 Similarity Score: {result.get('similarity_score', 0) * 100:.1f}%")
# #     if semantic_sim is not None:
# #         detail_cols[3].caption(f"🧠 Semantic (self): {semantic_sim * 100:.1f}%")

# #     if result.get("reference_similarity", 0) > 0 or result.get("best_reference_match"):
# #         st.markdown("**📚 Reference Corpus Match**")
# #         st.caption(f"Closest similarity found: {result['reference_similarity'] * 100:.1f}%")
# #         if result.get("best_reference_match"):
# #             st.markdown(f'<div class="ref-quote">"{result["best_reference_match"]}"</div>', unsafe_allow_html=True)
# #         if not result.get("grounded", False) and "Style-Flagged" in result.get("label", ""):
# #             st.warning("⚠️ No real source matched closely — verdict downgraded from Plagiarized to Unconfirmed.")

# #     st.markdown('</div>', unsafe_allow_html=True)


# # def status_header():
# #     c1, c2, c3, c4 = st.columns(4)
# #     model_ok = detector.is_loaded
# #     model_label = "Fine-Tuned (92.5%)" if detector.is_fine_tuned else ("Base Model" if model_ok else "Not Loaded")
# #     c1.markdown(
# #         f'<span class="status-pill {"pill-ok" if model_ok else "pill-bad"}">'
# #         f'{"✅" if model_ok else "❌"} Model: {model_label}</span>',
# #         unsafe_allow_html=True
# #     )
# #     c2.markdown(
# #         f'<span class="status-pill {"pill-ok" if humanizer.available else "pill-bad"}">'
# #         f'{"✅" if humanizer.available else "❌"} Gemini API</span>',
# #         unsafe_allow_html=True
# #     )
# #     c3.markdown(
# #         f'<span class="status-pill pill-ok">📚 {len(st.session_state.reference_texts)} Reference Docs</span>',
# #         unsafe_allow_html=True
# #     )
# #     c4.markdown(
# #         f'<span class="status-pill pill-ok">💻 {detector.device}</span>',
# #         unsafe_allow_html=True
# #     )
# #     if "_detector_load_error" in st.session_state:
# #         st.error(f"Model failed to load: {st.session_state['_detector_load_error']}")


# # # ======================================================================
# # # Pages
# # # ======================================================================
# # def page_home():
# #     st.title("📚 HUMA PLAG")
# #     st.caption("Plagiarism Detection & AI Text Humanization — powered by Hugging Face Longformer + Gemini")
# #     status_header()
# #     st.divider()

# #     st.markdown("""
# # Use the sidebar to navigate:

# # - **🔍 Check Plagiarism** — analyze a single piece of text
# # - **📊 Compare Two Texts** — direct similarity between two texts
# # - **✍️ Humanize Text** — rewrite text with Gemini in a chosen style
# # - **🔄 Full Pipeline** — detect, then automatically humanize
# # - **📦 Batch Check** — check many texts at once (paste or upload CSV)
# # - **📚 Reference Corpus** — manage the documents plagiarism verdicts are grounded against
# # - **📈 Performance & Stats** — model performance and session statistics
# # """)

# #     if st.session_state.history:
# #         st.divider()
# #         st.subheader("🕘 Recent Activity")
# #         for entry in reversed(st.session_state.history[-5:]):
# #             ts = entry["timestamp"].split("T")[1][:8]
# #             st.caption(f"`{ts}` — {entry['type'].replace('_', ' ').title()}")


# # def page_check_plagiarism():
# #     st.title("🔍 Check Text for Plagiarism")
# #     st.caption(
# #         "Runs the fine-tuned Longformer classifier, then grounds any 'Plagiarized' "
# #         "verdict against your reference corpus so a confident-but-groundless guess "
# #         "never gets shown as a hard fact."
# #     )

# #     text = st.text_area("Enter text to check (min 10 characters)", height=160, key="detect_text")
# #     analyze = st.button("🔍 Analyze", type="primary")

# #     if analyze:
# #         if len(text.strip()) < 10:
# #             st.warning("Please enter at least 10 characters.")
# #             return
# #         with st.spinner("Analyzing..."):
# #             result = detector.predict_grounded(text, st.session_state.reference_texts)
# #             semantic_sim = detector.calculate_semantic_similarity(text, text)
# #         log_history("plagiarism_check", text=text, result=result)
# #         render_detection_result(result, semantic_sim)


# # def page_compare_texts():
# #     st.title("📊 Compare Two Texts")
# #     st.caption("Direct cosine + semantic similarity between two texts, independent of the classifier.")

# #     col1, col2 = st.columns(2)
# #     with col1:
# #         text1 = st.text_area("First text", height=160, key="cmp_text1")
# #     with col2:
# #         text2 = st.text_area("Second text", height=160, key="cmp_text2")

# #     if st.button("📊 Compare", type="primary"):
# #         if len(text1.strip()) < 5 or len(text2.strip()) < 5:
# #             st.warning("Please enter at least 5 characters for each text.")
# #             return
# #         with st.spinner("Comparing..."):
# #             report = detector.compare_texts(text1, text2)
# #         log_history("compare_texts", text1=text1, text2=text2, report=report.to_dict())

# #         st.markdown('<div class="result-card">', unsafe_allow_html=True)
# #         c1, c2, c3 = st.columns(3)
# #         c1.metric("Cosine Similarity", f"{report.cosine_similarity * 100:.2f}%")
# #         c2.metric("Semantic Similarity", f"{report.semantic_similarity * 100:.2f}%")
# #         with c3:
# #             st.markdown("**Risk Level**")
# #             st.markdown(risk_badge_html(report.risk_level), unsafe_allow_html=True)
# #         st.caption(f"🕒 {report.timestamp}")
# #         st.markdown('</div>', unsafe_allow_html=True)


# # def page_humanize():
# #     st.title("✍️ Humanize Text with Gemini")

# #     if not humanizer.available:
# #         st.error("Gemini API not available. Set `GEMINI_API_KEY` in your `.env` file and restart the app.")
# #         return

# #     text = st.text_area("Text to humanize (min 10 characters)", height=160, key="hum_text")

# #     style_display = {
# #         "academic": "🎓 Academic (formal)",
# #         "casual": "💬 Casual (conversational)",
# #         "professional": "💼 Professional (business)",
# #         "creative": "🎨 Creative (engaging)",
# #         "simple": "📖 Simple (readable)",
# #     }
# #     style_key = st.selectbox("Style", list(style_display.keys()), format_func=lambda k: style_display[k])

# #     if st.button("✨ Humanize", type="primary"):
# #         if len(text.strip()) < 10:
# #             st.warning("Please enter at least 10 characters.")
# #             return
# #         with st.spinner("Humanizing with Gemini..."):
# #             result = humanizer.humanize(
# #                 text, style_key, similarity_calculator=detector.calculate_semantic_similarity
# #             )

# #         if result.get("success"):
# #             log_history("humanization", original_text=text, result=result)
# #             st.markdown("### ✅ Humanized Text")
# #             st.markdown(f'<div class="result-card">{result["humanized_text"]}</div>', unsafe_allow_html=True)

# #             c1, c2, c3, c4 = st.columns(4)
# #             c1.metric("Word Count", result.get("word_count", 0))
# #             c2.metric("Response Time", f"{result.get('response_time', 0):.2f}s")
# #             if "similarity_to_original" in result:
# #                 c3.metric("Similarity to Original", f"{result['similarity_to_original'] * 100:.1f}%")
# #             if "humanization_effective" in result:
# #                 c4.metric("Effective", "✅ Yes" if result["humanization_effective"] else "⚠️ No")
# #         else:
# #             st.error(f"Humanization failed: {result.get('error', 'Unknown error')}")


# # def page_full_pipeline():
# #     st.title("🔄 Full Pipeline — Detect + Humanize")
# #     st.caption("Runs plagiarism detection first, then always runs Gemini humanization on the same text.")

# #     text = st.text_area("Enter text", height=160, key="pipeline_text")

# #     if st.button("🚀 Run Pipeline", type="primary"):
# #         if len(text.strip()) < 10:
# #             st.warning("Please enter at least 10 characters.")
# #             return

# #         st.markdown("## Step 1 — Plagiarism Detection")
# #         with st.spinner("Detecting..."):
# #             detection = detector.predict_grounded(text, st.session_state.reference_texts)
# #         render_detection_result(detection)

# #         st.markdown("## Step 2 — Humanization")
# #         if humanizer.available:
# #             with st.spinner("Humanizing with Gemini..."):
# #                 humanized = humanizer.humanize(
# #                     text, "academic", similarity_calculator=detector.calculate_semantic_similarity
# #                 )
# #             if humanized.get("success"):
# #                 st.markdown(f'<div class="result-card">{humanized["humanized_text"]}</div>', unsafe_allow_html=True)
# #                 c1, c2 = st.columns(2)
# #                 c1.metric("Similarity to Original", f"{humanized.get('similarity_to_original', 0) * 100:.1f}%")
# #                 if "similarity_change" in humanized:
# #                     c2.metric("Similarity Change", f"{humanized['similarity_change'] * 100:+.1f}%")
# #             else:
# #                 st.error(f"Humanization failed: {humanized.get('error', 'Unknown error')}")
# #         else:
# #             st.warning("Gemini API not available for humanization.")

# #         log_history("full_pipeline", text=text, detection=detection)


# # def page_batch_check():
# #     st.title("📦 Batch Plagiarism Check")
# #     st.caption("Check many texts at once — paste one per line, or upload a CSV with a `text` column.")

# #     tab1, tab2 = st.tabs(["📋 Paste Text", "📁 Upload CSV"])

# #     texts = []
# #     with tab1:
# #         bulk_text = st.text_area("One text per line", height=180, key="batch_paste")
# #         if bulk_text.strip():
# #             texts = [line.strip() for line in bulk_text.split("\n") if line.strip()]
# #         run_paste = st.button("🔍 Analyze Pasted Texts", type="primary", key="batch_paste_btn")

# #     with tab2:
# #         uploaded = st.file_uploader("CSV file with a 'text' column", type=["csv"])
# #         run_csv = st.button("🔍 Analyze CSV", type="primary", key="batch_csv_btn")
# #         if uploaded is not None and run_csv:
# #             try:
# #                 df_in = pd.read_csv(uploaded)
# #                 if "text" not in df_in.columns:
# #                     st.error("CSV must contain a column named 'text'.")
# #                     texts = []
# #                 else:
# #                     texts = df_in["text"].dropna().astype(str).tolist()
# #             except Exception as e:
# #                 st.error(f"Could not read CSV: {e}")
# #                 texts = []

# #     should_run = (run_paste and texts) or (run_csv and uploaded is not None and texts)

# #     if should_run:
# #         if not texts:
# #             st.warning("No texts found to analyze.")
# #             return

# #         rows = []
# #         progress = st.progress(0.0, text=f"Processing 0/{len(texts)}...")
# #         for i, t in enumerate(texts):
# #             if len(t.strip()) >= 10:
# #                 result = detector.predict_grounded(t, st.session_state.reference_texts)
# #                 rows.append({
# #                     "Text": t[:100] + ("..." if len(t) > 100 else ""),
# #                     "Status": result.get("label", "Unknown"),
# #                     "Confidence": f"{result.get('confidence', 0) * 100:.1f}%",
# #                     "Risk Level": result.get("risk_level", "Unknown"),
# #                     "Reference Match": f"{result.get('reference_similarity', 0) * 100:.1f}%",
# #                 })
# #             else:
# #                 rows.append({
# #                     "Text": t, "Status": "Skipped (too short)", "Confidence": "-",
# #                     "Risk Level": "-", "Reference Match": "-"
# #                 })
# #             progress.progress((i + 1) / len(texts), text=f"Processing {i + 1}/{len(texts)}...")

# #         progress.empty()
# #         result_df = pd.DataFrame(rows)
# #         log_history("batch_check", total=len(texts))

# #         st.markdown(f"### Results — {len(texts)} texts processed")
# #         st.dataframe(result_df, width="stretch", hide_index=True)

# #         csv_bytes = result_df.to_csv(index=False).encode("utf-8")
# #         st.download_button("⬇️ Download Results as CSV", data=csv_bytes,
# #                             file_name="batch_plagiarism_results.csv", mime="text/csv")


# # def page_reference_corpus():
# #     st.title("📚 Reference Corpus")
# #     st.caption(
# #         "Documents in this corpus are what a 'Plagiarized' verdict is checked against. "
# #         "Without a real matching document here, a verdict gets downgraded to "
# #         "'Style-Flagged (No Matching Source)' instead of shown as a hard Plagiarized result."
# #     )

# #     st.info(f"📂 Corpus file: `{DEFAULT_CORPUS_PATH}` — {len(st.session_state.reference_texts)} documents loaded")

# #     with st.expander("➕ Add a reference document", expanded=False):
# #         new_doc = st.text_area("New reference document text", height=100, key="new_ref_doc")
# #         if st.button("Add to Corpus"):
# #             if len(new_doc.strip()) < 5:
# #                 st.warning("Please enter at least 5 characters.")
# #             else:
# #                 st.session_state.reference_texts.append(new_doc.strip())
# #                 try:
# #                     os.makedirs(os.path.dirname(DEFAULT_CORPUS_PATH) or ".", exist_ok=True)
# #                     with open(DEFAULT_CORPUS_PATH, "a", encoding="utf-8") as f:
# #                         f.write(new_doc.strip().replace("\n", " ") + "\n")
# #                     st.success("Added to corpus and saved to file.")
# #                     st.rerun()
# #                 except Exception as e:
# #                     st.error(f"Added in-session, but failed to save to file: {e}")

# #     st.markdown("### Current Documents")
# #     if not st.session_state.reference_texts:
# #         st.warning("No reference documents loaded. Add some above.")
# #     else:
# #         for i, doc in enumerate(st.session_state.reference_texts):
# #             col1, col2 = st.columns([10, 1])
# #             col1.markdown(f'<div class="ref-quote">{doc}</div>', unsafe_allow_html=True)
# #             if col2.button("🗑️", key=f"del_ref_{i}"):
# #                 st.session_state.reference_texts.pop(i)
# #                 try:
# #                     with open(DEFAULT_CORPUS_PATH, "w", encoding="utf-8") as f:
# #                         f.write("\n".join(st.session_state.reference_texts) + "\n")
# #                 except Exception as e:
# #                     st.error(f"Removed in-session, but failed to save to file: {e}")
# #                 st.rerun()

# #     if st.button("🔄 Reload Corpus from File"):
# #         st.session_state.reference_texts = load_reference_corpus()
# #         st.rerun()


# # def page_performance_stats():
# #     st.title("📈 Performance & Statistics")
# #     status_header()
# #     st.divider()

# #     st.subheader("🎯 Model Performance")
# #     if detector.is_loaded:
# #         metrics = detector.get_performance_metrics()
# #         c1, c2, c3, c4 = st.columns(4)
# #         c1.metric("Model", "Fine-Tuned" if metrics["is_fine_tuned"] else "Base")
# #         c2.metric("Predictions Made", metrics.get("total_predictions", 0))
# #         c3.metric("Avg Inference Time", f"{metrics.get('avg_inference_time', 0) * 1000:.0f} ms")
# #         c4.metric("Parameters", f"{metrics.get('model_parameters', 0):,}")

# #         eval_results = detector.results.get(metrics.get("model_name"), {})
# #         if eval_results:
# #             st.markdown("**Evaluation Metrics**")
# #             e1, e2, e3, e4 = st.columns(4)
# #             e1.metric("Accuracy", f"{eval_results.get('accuracy', 0) * 100:.2f}%")
# #             e2.metric("F1 Score", f"{eval_results.get('f1_score', 0):.4f}")
# #             e3.metric("Precision", f"{eval_results.get('precision', 0):.4f}")
# #             e4.metric("Recall", f"{eval_results.get('recall', 0):.4f}")
# #     else:
# #         st.warning("Model not loaded.")

# #     st.divider()
# #     st.subheader("✍️ Humanization Stats")
# #     hstats = humanizer.get_statistics()
# #     if hstats.get("total_humanizations", 0) > 0:
# #         c1, c2, c3 = st.columns(3)
# #         c1.metric("Total Humanizations", hstats["total_humanizations"])
# #         c2.metric("Success Rate", f"{hstats['success_rate']:.1f}%")
# #         c3.metric("Avg Response Time", f"{hstats['avg_response_time']:.2f}s")
# #         st.caption(f"Styles used: {', '.join(hstats.get('styles_used', [])) or '—'}")
# #     else:
# #         st.caption("No humanizations run yet this session.")

# #     st.divider()
# #     st.subheader("🕘 Session Activity")
# #     if st.session_state.history:
# #         counts = pd.Series([h["type"] for h in st.session_state.history]).value_counts()
# #         st.bar_chart(counts)
# #         with st.expander("Raw session history"):
# #             st.json(st.session_state.history)
# #     else:
# #         st.caption("No activity yet — run a check, comparison, or humanization to see stats here.")


# # # ======================================================================
# # # Sidebar navigation
# # # ======================================================================
# # with st.sidebar:
# #     st.markdown("## 📚 HUMA PLAG")
# #     st.caption("Plagiarism Detection & AI Humanization")
# #     st.divider()

# #     model_badge = "✅ Fine-Tuned" if detector.is_fine_tuned else ("✅ Base" if detector.is_loaded else "❌ Not Loaded")
# #     st.markdown(f"**Model:** {model_badge}")
# #     st.markdown(f"**Gemini:** {'✅ Available' if humanizer.available else '❌ Not Available'}")
# #     st.markdown(f"**Reference Docs:** {len(st.session_state.reference_texts)}")
# #     st.divider()

# #     page = st.radio(
# #         "Navigate",
# #         [
# #             "🏠 Home",
# #             "🔍 Check Plagiarism",
# #             "📊 Compare Two Texts",
# #             "✍️ Humanize Text",
# #             "🔄 Full Pipeline",
# #             "📦 Batch Check",
# #             "📚 Reference Corpus",
# #             "📈 Performance & Stats",
# #         ],
# #         label_visibility="collapsed",
# #     )

# # PAGES = {
# #     "🏠 Home": page_home,
# #     "🔍 Check Plagiarism": page_check_plagiarism,
# #     "📊 Compare Two Texts": page_compare_texts,
# #     "✍️ Humanize Text": page_humanize,
# #     "🔄 Full Pipeline": page_full_pipeline,
# #     "📦 Batch Check": page_batch_check,
# #     "📚 Reference Corpus": page_reference_corpus,
# #     "📈 Performance & Stats": page_performance_stats,
# # }

# # PAGES[page]()





"""
Streamlit UI - HUMA PLAG: Professional Plagiarism Detection
Stable Horizontal Top Navigation - Fixed & Polished
"""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

# --- Path setup, mirrors main.py ---------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.preprocessing import TextPreprocessor
from src.huggingface_detector import HuggingFaceDetector
from src.text_humanizer import GeminiHumanizer
from src.reference_corpus import load_reference_corpus, DEFAULT_CORPUS_PATH

try:
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv(dotenv_path=".env", encoding="utf-8")
except ImportError:
    pass

# ======================================================================
# Page config
# ======================================================================
st.set_page_config(
    page_title="HUMA PLAG - Professional Plagiarism Detection",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ======================================================================
# Global CSS
# ======================================================================
st.markdown("""
<style>
    .main { background: #f0f2f5; }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1300px;
        margin: 0 auto;
    }

    /* ===== TOP NAV BRAND STRIP ===== */
    .top-nav-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 0.9rem 2rem;
        margin: -1rem -1rem 1rem -1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.75rem;
        border-bottom: 3px solid #4a6cf7;
    }

    .nav-brand { display: flex; align-items: center; gap: 12px; }
    .nav-brand-icon { font-size: 1.8rem; }
    .nav-brand-text { font-size: 1.3rem; font-weight: 700; color: white; letter-spacing: -0.5px; }
    .nav-brand-text span { color: #4a6cf7; }
    .nav-brand-sub { font-size: 0.7rem; color: #8892b0; margin-left: 6px; font-weight: 400; }

    .nav-status { display: flex; align-items: center; gap: 1.25rem; }
    .nav-status-item { display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: #cbd3e6; }
    .nav-status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .nav-status-dot.green { background: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5); }
    .nav-status-dot.red { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.5); }

    /* ===== REAL NAV BUTTON ROW ===== */
    div[data-testid="stHorizontalBlock"].nav-row {
        background: #16213e;
        border-radius: 10px;
        padding: 6px;
        margin-bottom: 1.5rem;
    }

    .nav-row .stButton button {
        background: transparent !important;
        color: #b7c0da !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 0.4rem !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    .nav-row .stButton button:hover {
        background: rgba(74,108,247,0.18) !important;
        color: white !important;
        transform: none !important;
    }
    .nav-row .stButton button[kind="primary"] {
        background: #4a6cf7 !important;
        color: white !important;
        box-shadow: 0 3px 10px rgba(74,108,247,0.35) !important;
    }

    /* ===== PAGE CONTENT ===== */
    .page-content { padding: 0 0.25rem; }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
    }
    .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
    .card-title { font-size: 1.2rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem; }
    .card-subtitle { color: #6b7280; font-size: 0.9rem; }

    .risk-badge {
        display: inline-block;
        padding: 4px 18px;
        border-radius: 50px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
        height: 100%;
    }
    .metric-box:hover { border-color: #4a6cf7; box-shadow: 0 2px 8px rgba(74,108,247,0.08); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }

    /* Page-content action buttons keep the branded blue look */
    .page-content .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.8rem !important;
        background: #4a6cf7 !important;
        color: white !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .page-content .stButton button:hover {
        background: #3a5cd9 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(74,108,247,0.3) !important;
    }
    .page-content .stButton button:active { transform: translateY(0); }

    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        padding: 0.8rem !important;
        background: white !important;
    }
    .stTextArea textarea:focus {
        border-color: #4a6cf7 !important;
        box-shadow: 0 0 0 3px rgba(74,108,247,0.1) !important;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #4a6cf7, #6b8cff) !important;
        border-radius: 50px !important;
    }

    .ref-quote {
        border-left: 3px solid #4a6cf7;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        background: #f8fafc;
        border-radius: 4px;
        font-style: italic;
        color: #374151;
        font-size: 0.9rem;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 0.5rem 1.5rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #4a6cf7; color: white; }

    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6b7280;
        font-size: 0.8rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }

    @media (max-width: 768px) {
        .top-nav-bar { flex-direction: column; text-align: center; }
        .nav-row .stButton button { font-size: 0.7rem !important; padding: 0.4rem 0.2rem !important; }
        .card { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    "Low Risk": "#10b981",
    "Moderate Risk": "#f59e0b",
    "High Risk": "#f97316",
    "Very High Risk": "#ef4444",
    "Critical Risk": "#991b1b",
    "Uncertain": "#6b7280",
    "Unconfirmed": "#a16207",
    "No Data": "#6b7280",
    "Error": "#6b7280",
    "Unknown": "#6b7280",
}


def risk_badge_html(risk_level: str) -> str:
    color = RISK_COLORS.get(risk_level, "#6b7280")
    return f'<span class="risk-badge" style="background:{color};">{risk_level}</span>'


# ======================================================================
# Cached Resource Loaders
# ======================================================================
@st.cache_resource(show_spinner="Loading text preprocessor...")
def get_preprocessor():
    return TextPreprocessor()


@st.cache_resource(show_spinner="Loading plagiarism detection model...")
def get_detector():
    detector = HuggingFaceDetector()
    fine_tuned_path = os.path.join(CURRENT_DIR, "models", "fine_tuned_model")
    try:
        if os.path.exists(fine_tuned_path):
            detector.load_fine_tuned_model(fine_tuned_path)
        else:
            detector.load_model()
    except Exception as e:
        st.session_state["_detector_load_error"] = str(e)
    return detector


@st.cache_resource(show_spinner="Connecting to Gemini API...")
def get_humanizer():
    return GeminiHumanizer()


preprocessor = get_preprocessor()
detector = get_detector()
humanizer = get_humanizer()

if "reference_texts" not in st.session_state:
    st.session_state.reference_texts = load_reference_corpus()
if "history" not in st.session_state:
    st.session_state.history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

if "_detector_load_error" in st.session_state:
    st.warning(
        f"⚠️ The detection model failed to load: {st.session_state['_detector_load_error']}. "
        "Plagiarism checks won't work until this is resolved."
    )


def log_history(entry_type: str, **kwargs):
    st.session_state.history.append({
        "type": entry_type,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    })


def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()


# ======================================================================
# Render Helpers
# ======================================================================
def render_detection_result(result: dict, semantic_sim: float = None):
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**Status:** {result.get('label', 'Unknown')}")
    with col2:
        st.markdown(f"**Confidence:** {result.get('confidence', 0) * 100:.1f}%")
    with col3:
        st.markdown(risk_badge_html(result.get("risk_level", "Unknown")), unsafe_allow_html=True)

    st.divider()

    cols = st.columns(4)
    metrics = [
        (result.get("word_count", 0), "Word Count", ""),
        (result.get("probability", [0, 0])[1] * 100, "Plagiarism Score", "%"),
        (result.get("similarity_score", 0) * 100, "Similarity", "%"),
    ]
    for col, (val, label, suffix) in zip(cols, metrics):
        with col:
            display_val = f"{val:.1f}{suffix}" if isinstance(val, float) else val
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{display_val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    with cols[3]:
        if semantic_sim is not None:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{semantic_sim * 100:.1f}%</div>
                <div class="metric-label">Semantic Score</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-box"><div class="metric-value">—</div>'
                         '<div class="metric-label">Semantic Score</div></div>', unsafe_allow_html=True)

    st.progress(
        min(max(result.get("probability", [0, 0])[1], 0.0), 1.0),
        text=f"Plagiarism Probability: {result.get('probability', [0, 0])[1] * 100:.1f}%"
    )

    if result.get("reference_similarity", 0) > 0 or result.get("best_reference_match"):
        st.markdown("**Reference Corpus Match**")
        st.caption(f"Closest similarity: {result['reference_similarity'] * 100:.1f}%")
        if result.get("best_reference_match"):
            st.markdown(f'<div class="ref-quote">"{result["best_reference_match"]}"</div>', unsafe_allow_html=True)
        if not result.get("grounded", False) and "Style-Flagged" in result.get("label", ""):
            st.warning("No real source matched — verdict downgraded to Unconfirmed.")

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================================
# Top Navigation — brand strip + a real, clickable button row
# ======================================================================
NAV_ITEMS = [
    "🏠 Home",
    "🔍 Check Plagiarism",
    "📊 Compare Texts",
    "✍️ Humanize Text",
    "🔄 Full Pipeline",
    "📦 Batch Check",
    "📚 Reference Corpus",
    "📈 Performance & Stats",
]


def render_top_nav():
    current = st.session_state.current_page
    model_ok = detector.is_loaded
    gemini_ok = humanizer.available

    brand_html = f'''
    <div class="top-nav-bar">
        <div class="nav-brand">
            <span class="nav-brand-icon">📚</span>
            <span class="nav-brand-text">HUMA <span>PLAG</span><span class="nav-brand-sub">Professional</span></span>
        </div>
        <div class="nav-status">
            <span class="nav-status-item">
                <span class="nav-status-dot {'green' if model_ok else 'red'}"></span>
                {'Model OK' if model_ok else 'Model Offline'}
            </span>
            <span class="nav-status-item">
                <span class="nav-status-dot {'green' if gemini_ok else 'red'}"></span>
                {'Gemini OK' if gemini_ok else 'Gemini Offline'}
            </span>
            <span class="nav-status-item">
                <span class="nav-status-dot green"></span>
                {len(st.session_state.reference_texts)} Docs
            </span>
        </div>
    </div>
    '''
    st.markdown(brand_html, unsafe_allow_html=True)

    # A single real, visible, clickable row of nav buttons styled as pills.
    nav_container = st.container()
    with nav_container:
        cols = st.columns(len(NAV_ITEMS))
        for idx, item in enumerate(NAV_ITEMS):
            with cols[idx]:
                if st.button(
                    item,
                    key=f"nav_{idx}",
                    use_container_width=True,
                    type="primary" if item == current else "secondary",
                ):
                    navigate_to(item)

    # Tag this specific row so the dark-pill CSS only targets the nav,
    # not every button row in the app.
    st.markdown("""
    <script>
    const blocks = window.parent.document.querySelectorAll('div[data-testid="stHorizontalBlock"]');
    if (blocks.length > 0) { blocks[0].classList.add('nav-row'); }
    </script>
    """, unsafe_allow_html=True)


# ======================================================================
# Page Functions
# ======================================================================
def page_home():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <h1 style="font-size: 2rem; margin-bottom: 0.3rem; color: #1a1a2e;">Welcome to HUMA PLAG</h1>
        <p style="color: #4a5568; font-size: 1.1rem;">Professional Plagiarism Detection & AI Text Humanization</p>
        <p style="color: #6b7280;">Powered by Hugging Face Longformer + Google Gemini AI</p>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #4a6cf7; color: white; border-radius: 12px; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem;">✨</div>
            <div style="font-weight: 600;">Enterprise Grade</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">AI-Powered Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        (len([h for h in st.session_state.history if h["type"] == "plagiarism_check"]), "Checks Done"),
        (len([h for h in st.session_state.history if h["type"] == "humanization"]), "Humanizations"),
        (len([h for h in st.session_state.history if h["type"] == "compare_texts"]), "Comparisons"),
        (len(st.session_state.history), "Total Actions"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Quick Access")
    col1, col2, col3 = st.columns(3)
    quick_links = [
        (col1, "🔍", "Plagiarism Check", "Analyze text for plagiarism", "Go to Detection", "quick_detect", "🔍 Check Plagiarism"),
        (col2, "✍️", "Humanize Text", "Rewrite with AI", "Go to Humanize", "quick_humanize", "✍️ Humanize Text"),
        (col3, "📦", "Batch Check", "Analyze multiple texts", "Go to Batch", "quick_batch", "📦 Batch Check"),
    ]
    for col, icon, title, desc, btn_label, btn_key, target in quick_links:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div style="font-size: 2.5rem;">{icon}</div>
                <h4>{title}</h4>
                <p style="color: #6b7280; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(btn_label, key=btn_key, use_container_width=True):
                navigate_to(target)

    if st.session_state.history:
        st.divider()
        st.markdown("### Recent Activity")
        for entry in reversed(st.session_state.history[-5:]):
            ts = entry["timestamp"].split("T")[1][:8]
            action = entry["type"].replace("_", " ").title()
            st.caption(f"`{ts}` — {action}")


def page_check_plagiarism():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Check Plagiarism")
    st.markdown('<span class="card-subtitle">Analyze text for potential plagiarism with AI</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    text = st.text_area(
        "Enter text to analyze",
        height=180,
        key="detect_text",
        placeholder="Paste your text here (minimum 10 characters)..."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze:
        if len(text.strip()) < 10:
            st.warning("Please enter at least 10 characters.")
            return
        if not detector.is_loaded:
            st.error("The detection model isn't loaded, so this check can't run right now.")
            return

        with st.spinner("Analyzing text..."):
            result = detector.predict_grounded(text, st.session_state.reference_texts)
            semantic_sim = detector.calculate_semantic_similarity(text, text)

        log_history("plagiarism_check", result=result)
        render_detection_result(result, semantic_sim)


def page_compare_texts():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Compare Two Texts")
    st.markdown('<span class="card-subtitle">Direct similarity comparison between two texts</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("Text A", height=160, key="cmp_text1", placeholder="Enter first text...")
    with col2:
        text2 = st.text_area("Text B", height=160, key="cmp_text2", placeholder="Enter second text...")

    if st.button("📊 Compare", type="primary"):
        if len(text1.strip()) < 5 or len(text2.strip()) < 5:
            st.warning("Please enter at least 5 characters for each text.")
            return

        with st.spinner("Comparing texts..."):
            report = detector.compare_texts(text1, text2)

        log_history("compare_texts", report=report.to_dict())

        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Cosine Similarity", f"{report.cosine_similarity * 100:.2f}%")
        col2.metric("Semantic Similarity", f"{report.semantic_similarity * 100:.2f}%")
        with col3:
            st.markdown(risk_badge_html(report.risk_level), unsafe_allow_html=True)
        st.caption(f"Timestamp: {report.timestamp}")
        st.markdown('</div>', unsafe_allow_html=True)


def page_humanize():
    if not humanizer.available:
        st.error("❌ Gemini API not available. Please set GEMINI_API_KEY in your .env file.")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✍️ Humanize Text")
    st.markdown('<span class="card-subtitle">Rewrite text with Gemini AI in different styles</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    text = st.text_area(
        "Text to humanize",
        height=160,
        key="hum_text",
        placeholder="Enter text to rewrite in a more human style..."
    )

    style_display = {
        "academic": "🎓 Academic (Formal)",
        "casual": "💬 Casual (Conversational)",
        "professional": "💼 Professional (Business)",
        "creative": "🎨 Creative (Engaging)",
        "simple": "📖 Simple (Readable)",
    }

    style = st.selectbox("Select Writing Style", list(style_display.keys()), format_func=lambda k: style_display[k])

    if st.button("✨ Humanize", type="primary"):
        if len(text.strip()) < 10:
            st.warning("Please enter at least 10 characters.")
            return

        with st.spinner("Humanizing with Gemini..."):
            result = humanizer.humanize(
                text, style, similarity_calculator=detector.calculate_semantic_similarity
            )

        if result.get("success"):
            log_history("humanization", result=result)
            st.markdown("### ✅ Humanized Result")
            st.markdown(f'<div class="card">{result["humanized_text"]}</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Word Count", result.get("word_count", 0))
            col2.metric("Response Time", f"{result.get('response_time', 0):.2f}s")
            if "similarity_to_original" in result:
                col3.metric("Similarity to Original", f"{result['similarity_to_original'] * 100:.1f}%")
        else:
            st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")


def page_full_pipeline():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔄 Full Pipeline")
    st.markdown('<span class="card-subtitle">Detect plagiarism then automatically humanize</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    text = st.text_area("Enter text", height=160, key="pipeline_text", placeholder="Text will be analyzed and then humanized...")

    if st.button("🚀 Run Pipeline", type="primary"):
        if len(text.strip()) < 10:
            st.warning("Please enter at least 10 characters.")
            return

        st.markdown("### Step 1: 🔍 Plagiarism Detection")
        with st.spinner("Detecting..."):
            detection = detector.predict_grounded(text, st.session_state.reference_texts)
        render_detection_result(detection)

        st.markdown("### Step 2: ✍️ Humanization")
        if humanizer.available:
            with st.spinner("Humanizing..."):
                humanized = humanizer.humanize(
                    text, "academic", similarity_calculator=detector.calculate_semantic_similarity
                )
            if humanized.get("success"):
                st.markdown(f'<div class="card">{humanized["humanized_text"]}</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.metric("Similarity to Original", f"{humanized.get('similarity_to_original', 0) * 100:.1f}%")
                if "similarity_change" in humanized:
                    col2.metric("Similarity Change", f"{humanized['similarity_change'] * 100:+.1f}%")
            else:
                st.error(f"Humanization failed: {humanized.get('error', 'Unknown error')}")
        else:
            st.warning("Gemini API not available.")

        log_history("full_pipeline", detection=detection)


def page_batch_check():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📦 Batch Check")
    st.markdown('<span class="card-subtitle">Analyze multiple texts at once</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Paste Text", "📁 Upload CSV"])

    texts = []
    with tab1:
        bulk_text = st.text_area("One text per line", height=180, key="batch_paste", placeholder="Text 1\nText 2\nText 3\n...")
        if bulk_text.strip():
            texts = [line.strip() for line in bulk_text.split("\n") if line.strip()]
        run_paste = st.button("🔍 Analyze Pasted Texts", type="primary", key="batch_paste_btn")

    with tab2:
        uploaded = st.file_uploader("CSV with 'text' column", type=["csv"])
        run_csv = st.button("🔍 Analyze CSV", type="primary", key="batch_csv_btn")
        if uploaded is not None and run_csv:
            try:
                df_in = pd.read_csv(uploaded)
                if "text" not in df_in.columns:
                    st.error("CSV must contain a 'text' column.")
                    texts = []
                else:
                    texts = df_in["text"].dropna().astype(str).tolist()
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                texts = []

    should_run = (run_paste and texts) or (run_csv and uploaded is not None and texts)

    if should_run:
        if not texts:
            st.warning("No texts found.")
            return

        rows = []
        progress = st.progress(0, text=f"Processing 0/{len(texts)}...")

        for i, t in enumerate(texts):
            if len(t.strip()) >= 10:
                result = detector.predict_grounded(t, st.session_state.reference_texts)
                rows.append({
                    "Text": t[:100] + ("..." if len(t) > 100 else ""),
                    "Status": result.get("label", "Unknown"),
                    "Confidence": f"{result.get('confidence', 0) * 100:.1f}%",
                    "Risk Level": result.get("risk_level", "Unknown"),
                    "Reference Match": f"{result.get('reference_similarity', 0) * 100:.1f}%",
                })
            else:
                rows.append({
                    "Text": t, "Status": "Skipped (too short)",
                    "Confidence": "-", "Risk Level": "-", "Reference Match": "-"
                })
            progress.progress((i + 1) / len(texts), text=f"Processing {i + 1}/{len(texts)}...")

        progress.empty()
        result_df = pd.DataFrame(rows)
        log_history("batch_check", total=len(texts))

        st.markdown(f"### ✅ Results — {len(texts)} texts processed")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def page_reference_corpus():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📚 Reference Corpus")
    st.markdown(f'<span class="card-subtitle">Corpus: {DEFAULT_CORPUS_PATH} — {len(st.session_state.reference_texts)} documents</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("➕ Add Document", expanded=False):
        new_doc = st.text_area("Document text", height=100, key="new_ref_doc", placeholder="Enter reference text...")
        if st.button("Add to Corpus", type="primary"):
            if len(new_doc.strip()) < 5:
                st.warning("Please enter at least 5 characters.")
            else:
                st.session_state.reference_texts.append(new_doc.strip())
                try:
                    os.makedirs(os.path.dirname(DEFAULT_CORPUS_PATH) or ".", exist_ok=True)
                    with open(DEFAULT_CORPUS_PATH, "a", encoding="utf-8") as f:
                        f.write(new_doc.strip().replace("\n", " ") + "\n")
                    st.success("✅ Added successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Added in-session, but failed to save: {e}")

    st.markdown("### Current Documents")
    if not st.session_state.reference_texts:
        st.warning("No documents loaded. Add documents above.")
    else:
        for i, doc in enumerate(st.session_state.reference_texts):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f'<div class="ref-quote">{doc[:200]}{"..." if len(doc) > 200 else ""}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_ref_{i}"):
                    st.session_state.reference_texts.pop(i)
                    try:
                        with open(DEFAULT_CORPUS_PATH, "w", encoding="utf-8") as f:
                            f.write("\n".join(st.session_state.reference_texts) + "\n")
                    except Exception as e:
                        st.error(f"Removed in-session, but failed to save: {e}")
                    st.rerun()

    if st.button("🔄 Reload Corpus"):
        st.session_state.reference_texts = load_reference_corpus()
        st.rerun()


def page_performance_stats():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Performance & Statistics")
    st.markdown('<span class="card-subtitle">System performance metrics and analytics</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    model_ok = detector.is_loaded
    model_label = "Fine-Tuned" if detector.is_fine_tuned else ("Base" if model_ok else "Not Loaded")

    status_blocks = [
        (col1, model_ok, "Model", model_label),
        (col2, humanizer.available, "Gemini API", "Available" if humanizer.available else "Not Available"),
    ]
    for col, ok, title, label in status_blocks:
        with col:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 2rem;">{'✅' if ok else '❌'}</div>
                <div><strong>{title}</strong></div>
                <div style="color: {'#10b981' if ok else '#ef4444'};">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">📚</div>
            <div><strong>Reference Docs</strong></div>
            <div style="color: #10b981;">{len(st.session_state.reference_texts)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 2rem;">💻</div>
            <div><strong>Device</strong></div>
            <div style="color: #10b981;">{detector.device}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### Model Performance")
    if detector.is_loaded:
        metrics = detector.get_performance_metrics()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model Type", "Fine-Tuned" if metrics["is_fine_tuned"] else "Base")
        col2.metric("Total Predictions", metrics.get("total_predictions", 0))
        col3.metric("Avg Inference Time", f"{metrics.get('avg_inference_time', 0) * 1000:.0f} ms")
        col4.metric("Parameters", f"{metrics.get('model_parameters', 0):,}")

        model_name = metrics.get("model_name")
        eval_results = detector.results.get(model_name, {}) if model_name else {}
        if eval_results:
            st.markdown("#### Evaluation Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{eval_results.get('accuracy', 0) * 100:.2f}%")
            col2.metric("F1 Score", f"{eval_results.get('f1_score', 0):.4f}")
            col3.metric("Precision", f"{eval_results.get('precision', 0):.4f}")
            col4.metric("Recall", f"{eval_results.get('recall', 0):.4f}")
    else:
        st.warning("Model not loaded.")

    st.divider()

    st.markdown("### Humanization Stats")
    hstats = humanizer.get_statistics()
    if hstats.get("total_humanizations", 0) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Humanizations", hstats["total_humanizations"])
        col2.metric("Success Rate", f"{hstats['success_rate']:.1f}%")
        col3.metric("Avg Response Time", f"{hstats['avg_response_time']:.2f}s")
        if hstats.get("styles_used"):
            st.caption(f"Styles Used: {', '.join(hstats['styles_used'])}")
    else:
        st.caption("No humanizations run yet this session.")

    st.divider()

    st.markdown("### Session Activity")
    if st.session_state.history:
        counts = pd.Series([h["type"] for h in st.session_state.history]).value_counts()
        st.bar_chart(counts)
        with st.expander("Detailed Session History"):
            st.json(st.session_state.history)
    else:
        st.caption("No activity yet.")


# ======================================================================
# Main App
# ======================================================================
render_top_nav()

st.markdown('<div class="page-content">', unsafe_allow_html=True)

PAGE_FUNCTIONS = {
    "🏠 Home": page_home,
    "🔍 Check Plagiarism": page_check_plagiarism,
    "📊 Compare Texts": page_compare_texts,
    "✍️ Humanize Text": page_humanize,
    "🔄 Full Pipeline": page_full_pipeline,
    "📦 Batch Check": page_batch_check,
    "📚 Reference Corpus": page_reference_corpus,
    "📈 Performance & Stats": page_performance_stats,
}

current_page = st.session_state.current_page
if current_page in PAGE_FUNCTIONS:
    PAGE_FUNCTIONS[current_page]()

st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================
# Footer
# ======================================================================
st.markdown("""
<div class="footer">
    <strong>HUMA PLAG</strong> — Professional Plagiarism Detection & AI Text Humanization
    <br>
    Powered by Hugging Face Transformers & Google Gemini AI
</div>
""", unsafe_allow_html=True)


