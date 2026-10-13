"""
PaperLens: AI-Powered Research Paper Assistant
Version 2.0 (High-Performance Single Drop-In File)

An interactive Streamlit application designed to help students, academics,
and researchers deeply read, understand, summarize, and question academic papers.

Core Features:
1. Understand Paper: Two-pass guided reading report breaking down motivation,
   prerequisites, methodology, and conclusions.
2. Paper Summary: Revision-friendly structured synthesis matching the paper's
   original section headings.
3. Ask the Paper: Grounded question-answering powered by BM25 retrieval with
   page-level citations.
4. Word Explorer: Technical terminology breakdown with definitions, analogies,
   and in-paper usage context.
5. Deep Dive: Granular technical analysis of specific equations, experiments,
   or subsections.

How to Run:
1. Ensure your .env file in this directory contains:
   GROQ_API_KEY=gsk_your_actual_key_here
2. Install dependencies:
   pip install -r requirements.txt
3. Start the application:
   streamlit run app.py
"""

# ==============================================================================
# SECTION 1: Standard & Third-Party Imports
# ==============================================================================
import os
import re
import math
import time
from io import BytesIO
from typing import Dict, List, Tuple, Generator, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Load local environment variables from .env file (silently in background)
load_dotenv()


# ==============================================================================
# SECTION 2: BM25 Information Retrieval Engine (Pure Python, Zero Dependency)
# ==============================================================================
class BM25Retriever:
    """
    A lightweight, mathematically sound BM25Okapi search engine.
    
    Why BM25 over simple keyword counting?
    - Simple keyword search counts words like 'a', 'the', or matches substrings
      (e.g., 'cat' inside 'communication').
    - BM25 accounts for:
      1. Term Frequency (TF): How often a query word appears in a chunk.
      2. Inverse Document Frequency (IDF): Rare technical terms (e.g. 'backpropagation')
         get much higher weight than common academic words (e.g. 'paper', 'method').
      3. Document Length Normalization: Prevents longer chunks from artificially winning.
      4. Exact Phrase Boost: Rewards chunks where the query appears verbatim.
    """
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        """
        Initialize index with paper chunks.
        k1 (default 1.5): Controls term frequency saturation.
        b (default 0.75): Controls document length normalization penalty.
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.avg_dl = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.chunk_tokens: List[List[str]] = []

        # Common stop-words to ignore during scoring
        self.stopwords = {
            "the", "and", "for", "that", "this", "with", "from", "what",
            "when", "where", "which", "does", "into", "about", "are", "was",
            "were", "how", "why", "can", "could", "would", "should", "paper",
            "explain", "please", "tell", "describe", "give", "its", "their",
            "have", "has", "had", "not", "but", "you", "your", "use", "used",
            "then", "been", "than", "also", "each", "more", "most", "some",
            "such", "only", "other", "them", "these", "they", "will",
        }

        self._build_index()

    def tokenize(self, text: str) -> List[str]:
        """Convert text into lowercase individual words, stripping punctuation."""
        clean = text.lower()
        # Find alpha-numeric words of at least 2 characters
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", clean)
        return [w for w in words if w not in self.stopwords]

    def _build_index(self):
        """Pre-compute document lengths, inverted frequencies, and average length."""
        total_len = 0
        for chunk in self.chunks:
            tokens = self.tokenize(chunk["text"])
            self.chunk_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            # Track how many distinct chunks contain each term
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_dl = (total_len / self.corpus_size) if self.corpus_size > 0 else 1.0

    def get_idf(self, term: str) -> float:
        """Compute Robertson-Spärck Jones Inverse Document Frequency (IDF)."""
        df = self.doc_freqs.get(term, 0)
        # Smoothed BM25 IDF formula
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank all chunks against the query and return the top_k best matches.
        Returns a list of tuples: (chunk_dict, bm25_score).
        """
        if not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        # Fallback: if query is empty or only stopwords, return top chunks
        if not query_tokens:
            return [(chunk, 0.0) for chunk in self.chunks[:top_k]]

        clean_phrase = " ".join(query_tokens)
        scores: List[Tuple[float, int, Dict[str, Any]]] = []

        for idx, chunk in enumerate(self.chunks):
            tokens = self.chunk_tokens[idx]
            doc_len = self.doc_lens[idx]
            if doc_len == 0:
                continue

            # Calculate Term Frequency (TF) for this chunk
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            # Compute BM25 formula
            score = 0.0
            for term in query_tokens:
                if term in tf_map:
                    tf = tf_map[term]
                    idf = self.get_idf(term)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_dl))
                    score += idf * (numerator / denominator)

            # Exact phrase boost (reward verbatim matches by 30%)
            chunk_text_lower = chunk["text"].lower()
            if clean_phrase in chunk_text_lower:
                score *= 1.30

            if score > 0:
                scores.append((score, -idx, chunk))

        # Sort descending by score
        scores.sort(reverse=True)

        if scores:
            return [(item[2], item[0]) for item in scores[:top_k]]

        # Fallback if no keywords matched: return the introductory chunks
        return [(chunk, 0.0) for chunk in self.chunks[:min(2, len(self.chunks))]]


# ==============================================================================
# SECTION 3: Document Processing & PDF Text Ingestion
# ==============================================================================
def process_pdf(uploaded_file) -> Dict[str, Any]:
    """
    Parse uploaded PDF bytes into structured chunks.
    Key tasks:
    1. Extract page text using pypdf.
    2. De-hyphenate broken words across line breaks (e.g. 'trans-\nformer' -> 'transformer').
    3. Retain page number metadata on every excerpt so answers can cite exact pages.
    4. Split into manageable text chunks using RecursiveCharacterTextSplitter.
    """
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    if reader.is_encrypted:
        raise ValueError("This PDF is encrypted. Please provide an unencrypted PDF.")

    page_data: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = (page.extract_text() or "").strip()
        if raw_text:
            # Repair hyphenated word wraps common in academic double-column PDFs
            normalized_text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", raw_text)
            page_data.append({"page": page_idx, "text": normalized_text})
            full_text_parts.append(f"[Page {page_idx}]\n{normalized_text}")

    full_text = "\n\n".join(full_text_parts).strip()
    chunks: List[Dict[str, Any]] = []

    if full_text_parts:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,
            chunk_overlap=350,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunk_counter = 1
        for p in page_data:
            page_chunks = splitter.split_text(p["text"])
            for text_chunk in page_chunks:
                if len(text_chunk.strip()) > 40:
                    chunks.append({
                        "id": chunk_counter,
                        "page": p["page"],
                        "text": text_chunk.strip()
                    })
                    chunk_counter += 1

    # Extract or infer paper title from metadata
    metadata = reader.metadata
    title = metadata.title.strip() if (metadata and metadata.title) else ""
    if not title or len(title) < 4:
        title = uploaded_file.name

    return {
        "name": uploaded_file.name,
        "title": title,
        "pages": len(reader.pages),
        "pages_with_text": len(page_data),
        "text": full_text,
        "chunks": chunks,
        "processed": True,
    }


# ==============================================================================
# SECTION 4: AI Client, Retry Logic & Streaming Generation
# ==============================================================================
def get_groq_client(api_key: str) -> Optional[Groq]:
    """Create Groq client instance."""
    if not api_key:
        return None
    return Groq(api_key=api_key)

def ask_groq_text(
    client: Groq,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> str:
    """
    Synchronous API call with automatic exponential backoff retry.
    Used for intermediate map-pass steps when synthesizing multiple sections.
    """
    delay = 2.0
    last_err = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Model returned an empty response.")
            return content.strip()

        except Exception as e:
            last_err = e
            # Sleep with exponential backoff on rate limits or network issues
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2.0
            else:
                raise last_err

    raise last_err or RuntimeError("Failed after retries.")

def ask_groq_stream(
    client: Groq,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 3000,
) -> Generator[str, None, None]:
    """
    Streaming API call that yields tokens chunk-by-chunk in real time.
    Provides immediate visual response for Streamlit's st.write_stream().
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def display_retrieval_citations(results: List[Tuple[Dict[str, Any], float]]):
    """
    Render an interactive collapsible drawer displaying the exact
    paper chunks, page numbers, and relevance scores grounding the answer.
    """
    with st.expander(f"📚 Retrieved Evidence & Citations ({len(results)} chunks)", expanded=False):
        for chunk, score in results:
            st.markdown(
                f"""
                <div class="citation-box">
                    <div class="citation-meta">📄 Page {chunk['page']} · Chunk #{chunk['id']} · Relevance Score: {score:.2f}</div>
                    <div>{chunk['text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# SECTION 5: Main Application Entry Point
# ==============================================================================
def main():
    # --------------------------------------------------------------------------
    # 5.1 Page Configuration & Responsive Design
    # --------------------------------------------------------------------------
    st.set_page_config(
        page_title="PaperLens — AI Research Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling for hero section, cards, and citation boxes
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15);
        }
        .hero {
            padding: 1.8rem 2.2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(91, 100, 240, 0.14), rgba(36, 180, 170, 0.09));
            border: 1px solid rgba(128, 128, 128, 0.18);
            margin-bottom: 1.5rem;
        }
        .hero-badge {
            display: inline-block;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            background: rgba(91, 100, 240, 0.2);
            color: #5b64f0;
            margin-bottom: 0.6rem;
        }
        .feature-card {
            padding: 1.2rem;
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            min-height: 135px;
            margin-bottom: 0.8rem;
            background: rgba(128, 128, 128, 0.03);
        }
        .citation-box {
            padding: 0.8rem 1rem;
            border-left: 3px solid #5b64f0;
            background: rgba(91, 100, 240, 0.05);
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.6rem;
            font-size: 0.88rem;
        }
        .citation-meta {
            font-size: 0.76rem;
            font-weight: 600;
            opacity: 0.8;
            margin-bottom: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 5.2 Sidebar Setup (Clean, Secure & Minimalist)
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("# 📄 PaperLens")
        st.caption("YOUR AI RESEARCH ASSISTANT")
        st.divider()

        # Load API key silently from .env (no UI inputs or exposed passwords)
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("API key not found. Please ensure GROQ_API_KEY is defined in your .env file.")
            st.stop()

        client = get_groq_client(api_key)

        # Locked to the configured model and optimal temperature
        selected_model = "openai/gpt-oss-20b"
        temperature = 0.2

        st.markdown("### 🧭 Workspace Mode")

        features = {
            "Understand Paper": "🧠",
            "Paper Summary": "📝",
            "Ask the Paper": "💬",
            "Word Explorer": "🔎",
            "Deep Dive": "🧪",
        }

        if "active_feature" not in st.session_state:
            st.session_state.active_feature = "Understand Paper"

        st.radio(
            "WORKSPACE",
            options=list(features.keys()),
            format_func=lambda f: f"{features[f]}  {f}",
            key="active_feature",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### 📑 Current Document")

        paper = st.session_state.get("paper_data")

        if paper:
            st.markdown(f"**{paper['name']}**")
            st.caption(f"{paper['pages']} pages · {len(paper['chunks'])} chunks indexed")

            # Unload document and reset generated state
            if st.button("🗑️ Unload Document", use_container_width=True):
                st.session_state.paper_data = None
                st.session_state.retriever = None
                for key in ["understanding_report", "paper_summary", "paper_answers", "word_explanation", "deep_dive_result"]:
                    st.session_state[key] = None
                st.rerun()
        else:
            st.caption("No document loaded.")

        st.divider()

        # File upload accordion
        with st.expander("📤 Upload / Replace PDF", expanded=(paper is None)):
            uploaded_pdf = st.file_uploader(
                "Select research PDF",
                type=["pdf"],
                accept_multiple_files=False,
                key="pdf_uploader",
            )

            if st.button("⚡ Process PDF", use_container_width=True, disabled=uploaded_pdf is None):
                try:
                    with st.spinner("Extracting text and building BM25 index..."):
                        processed = process_pdf(uploaded_pdf)
                        st.session_state.paper_data = processed
                        st.session_state.retriever = BM25Retriever(processed["chunks"])

                        # Reset previous feature answers
                        for key in ["understanding_report", "paper_summary", "paper_answers", "word_explanation", "deep_dive_result"]:
                            st.session_state[key] = None

                    st.success("Document indexed with BM25 engine!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    # --------------------------------------------------------------------------
    # 5.3 Main Area Hero Header
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">RESEARCH ASSISTANT 2.0</div>
            <h1 style="margin: 0.2rem 0;">PaperLens</h1>
            <p style="margin: 0; opacity: 0.85;">Deep understanding, structured synthesis, and grounded retrieval for academic papers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 5.4 Empty State (Prompt user to upload a paper)
    # --------------------------------------------------------------------------
    paper = st.session_state.get("paper_data")
    retriever: Optional[BM25Retriever] = st.session_state.get("retriever")

    if paper is None:
        st.subheader("Welcome to PaperLens")
        st.write("Upload a research paper in the left sidebar to unlock the 5 study modes:")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="feature-card">
                    <h4>🧠 Understand Paper</h4>
                    <p>Guided, first-read report breaking down motivation, prerequisites, section-by-section logic, and findings.</p>
                </div>
                <div class="feature-card">
                    <h4>💬 Ask the Paper</h4>
                    <p>Pose specific questions answered with BM25-ranked passages and page-level source evidence.</p>
                </div>
                <div class="feature-card">
                    <h4>🧪 Deep Dive</h4>
                    <p>Rigorous breakdown of specific sections, methodology steps, mathematical formulations, or experiments.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="feature-card">
                    <h4>📝 Paper Summary</h4>
                    <p>Revision-ready synthesis matching the paper's original headings and core technical contributions.</p>
                </div>
                <div class="feature-card">
                    <h4>🔎 Word Explorer</h4>
                    <p>Deconstruct technical terminology and abbreviations with in-paper context and intuitive analogies.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.stop()

    # --------------------------------------------------------------------------
    # 5.5 Document Metrics Bar
    # --------------------------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Pages", paper["pages"])
    with col_m2:
        st.metric("Pages with Text", paper["pages_with_text"])
    with col_m3:
        st.metric("Indexed Chunks", len(paper["chunks"]))
    with col_m4:
        st.metric("Engine", selected_model.split("/")[-1])

    if not paper["chunks"]:
        st.error("No extractable text was found in this PDF. It may be a scanned document requiring OCR.")
        st.stop()

    st.divider()
    active_mode = st.session_state.active_feature

    # --------------------------------------------------------------------------
    # FEATURE 1: Understand Paper (Two-Pass Guided Reading Report)
    # --------------------------------------------------------------------------
    if active_mode == "Understand Paper":
        st.header("🧠 Understand Paper")
        st.caption("A structured, pedagogical companion designed for your first complete read.")
        st.info("Generates a two-pass synthesis: first analyzing discrete paper segments, then streaming an integrated pedagogical guide.")

        if st.button("🚀 Generate Guided Reading Report", type="primary", use_container_width=True):
            try:
                chunks = paper["chunks"]
                progress_bar = st.progress(0, text="Analyzing paper sections (Pass 1/2)...")

                # Pass 1: Map stage across chunk segments
                segment_size = max(1, len(chunks) // 4)
                segments = [chunks[i:i + segment_size] for i in range(0, len(chunks), segment_size)]
                intermediate_notes = []

                for idx, seg in enumerate(segments):
                    seg_text = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in seg)
                    progress_bar.progress(
                        int((idx / len(segments)) * 60),
                        text=f"Analyzing segment {idx + 1} of {len(segments)}...",
                    )
                    notes = ask_groq_text(
                        client=client,
                        model=selected_model,
                        system_prompt="You are PaperLens, an academic tutor. Identify main contributions, methodology, and key concepts without hallucinating.",
                        user_prompt=f"Analyze this paper segment:\n\n{seg_text}",
                        temperature=temperature,
                        max_tokens=1500,
                    )
                    intermediate_notes.append(notes)

                # Pass 2: Reduce stage with live token streaming
                progress_bar.progress(70, text="Synthesizing final reading report (Pass 2/2)...")

                combined_notes = "\n\n--- NEXT SEGMENT ---\n\n".join(intermediate_notes)
                final_prompt = f"""Create an accurate, structured guided reading report from these research notes:

1. Prerequisites & Background Needed
2. Core Motivation & Problem Statement
3. Section-by-Section Logical Flow
4. Methodology & Architectural Overview
5. Key Experimental Results & Benchmarks
6. Known Limitations & Future Directions

RESEARCH NOTES:
{combined_notes}"""

                st.markdown("### 📖 Guided Reading Report")
                stream_gen = ask_groq_stream(
                    client=client,
                    model=selected_model,
                    system_prompt="You are PaperLens. Produce a structured, highly educational, technically precise reading guide.",
                    user_prompt=final_prompt,
                    temperature=temperature,
                    max_tokens=4000,
                )
                report_text = st.write_stream(stream_gen)
                progress_bar.progress(100, text="Complete!")
                st.session_state["understanding_report"] = report_text

            except Exception as e:
                st.error(f"Generation error: {e}")

        # Download button for generated report
        saved_report = st.session_state.get("understanding_report")
        if saved_report:
            st.download_button(
                "💾 Download Report as Markdown",
                data=saved_report,
                file_name=f"{paper['name']}_understanding_report.md",
                mime="text/markdown",
            )

    # --------------------------------------------------------------------------
    # FEATURE 2: Paper Summary (Revision Notes)
    # --------------------------------------------------------------------------
    elif active_mode == "Paper Summary":
        st.header("📝 Paper Summary")
        st.caption("A concise, revision-ready summary strictly preserving the paper's original headings and contributions.")

        if st.button("🚀 Generate Revision Summary", type="primary", use_container_width=True):
            try:
                chunks = paper["chunks"]
                progress_bar = st.progress(0, text="Extracting key arguments...")

                segment_size = max(1, len(chunks) // 3)
                segments = [chunks[i:i + segment_size] for i in range(0, len(chunks), segment_size)]
                intermediate_notes = []

                for idx, seg in enumerate(segments):
                    seg_text = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in seg)
                    progress_bar.progress(int((idx / len(segments)) * 60))
                    notes = ask_groq_text(
                        client=client,
                        model=selected_model,
                        system_prompt="Summarize this section accurately for quick revision. Capture technical details, methods, and findings.",
                        user_prompt=f"Section text:\n\n{seg_text}",
                        temperature=temperature,
                        max_tokens=1200,
                    )
                    intermediate_notes.append(notes)

                progress_bar.progress(70, text="Generating final summary...")

                final_prompt = f"""Synthesize a revision summary from these section notes:
- Respect the paper's original headings and logical order.
- Highlight specific metrics, benchmarks, and claims.
- Conclude with stated limitations.

NOTES:
{"\n\n".join(intermediate_notes)}"""

                st.markdown("### 📋 Executive Revision Summary")
                stream_gen = ask_groq_stream(
                    client=client,
                    model=selected_model,
                    system_prompt="You are PaperLens. Deliver a polished, bulleted revision summary.",
                    user_prompt=final_prompt,
                    temperature=temperature,
                    max_tokens=3500,
                )
                summary_text = st.write_stream(stream_gen)
                progress_bar.progress(100, text="Complete!")
                st.session_state["paper_summary"] = summary_text

            except Exception as e:
                st.error(f"Generation error: {e}")

        saved_summary = st.session_state.get("paper_summary")
        if saved_summary:
            st.download_button(
                "💾 Download Summary as Markdown",
                data=saved_summary,
                file_name=f"{paper['name']}_summary.md",
                mime="text/markdown",
            )

    # --------------------------------------------------------------------------
    # FEATURE 3: Ask the Paper (BM25 Q&A with Citations)
    # --------------------------------------------------------------------------
    elif active_mode == "Ask the Paper":
        st.header("💬 Ask the Paper")
        st.caption("Ask questions grounded strictly in the text, powered by BM25 retrieval.")

        col_q, col_ex = st.columns([3, 1])
        with col_ex:
            st.markdown("**Suggested Questions:**")
            if st.button("Core Contribution?", use_container_width=True):
                st.session_state.paper_question = "What is the primary contribution or proposed method of this paper?"
            if st.button("Datasets Used?", use_container_width=True):
                st.session_state.paper_question = "What datasets, benchmarks, or experimental setup were used to evaluate the method?"
            if st.button("Key Limitations?", use_container_width=True):
                st.session_state.paper_question = "What are the primary limitations or failure modes acknowledged by the authors?"

        with col_q:
            question = st.text_area(
                "Enter your question:",
                value=st.session_state.get("paper_question", ""),
                placeholder="e.g., How does the attention mechanism in Section 3 differ from prior work?",
                height=110,
            )

        if st.button("🔍 Search & Answer", type="primary", disabled=not question.strip()):
            try:
                with st.spinner("Executing BM25 retrieval across document chunks..."):
                    retrieved = retriever.retrieve(question, top_k=5)

                # Show interactive citations drawer
                display_retrieval_citations(retrieved)

                context_str = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c, _ in retrieved)
                prompt = f"""Question: {question}

Context Excerpts from Paper:
{context_str}

Instructions:
Answer the question thoroughly using ONLY the supplied excerpts.
Cite relevant page numbers (e.g. '[Page X]') when referencing specific details.
If the excerpts do not contain enough information, clearly state what is missing."""

                st.markdown("### 💡 Answer")
                stream_gen = ask_groq_stream(
                    client=client,
                    model=selected_model,
                    system_prompt="You are PaperLens. Answer questions accurately based on supplied excerpts with explicit page citations.",
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=2000,
                )
                answer_text = st.write_stream(stream_gen)
                st.session_state["paper_answers"] = {"q": question, "a": answer_text, "sources": retrieved}

            except Exception as e:
                st.error(f"Error answering question: {e}")

        elif st.session_state.get("paper_answers"):
            last_qa = st.session_state["paper_answers"]
            if last_qa.get("sources"):
                display_retrieval_citations(last_qa["sources"])

    # --------------------------------------------------------------------------
    # FEATURE 4: Word Explorer (In-Context Technical Definitions)
    # --------------------------------------------------------------------------
    elif active_mode == "Word Explorer":
        st.header("🔎 Word Explorer")
        st.caption("Explain unfamiliar technical terminology, equations, or abbreviations in context.")

        term = st.text_input("Technical Term or Acronym:", placeholder="e.g., Latent Space, Perplexity, LoRA, Ablation").strip()

        if st.button("📖 Deconstruct Term", type="primary", disabled=not term):
            try:
                with st.spinner(f"Locating occurrences of '{term}'..."):
                    retrieved = retriever.retrieve(term, top_k=4)

                display_retrieval_citations(retrieved)

                context_str = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c, _ in retrieved)
                prompt = f"""Term: {term}

Paper Passages referencing or relating to the term:
{context_str}

Please explain this term using the following structure:
1. 💡 Simple Definition (Plain English, for a newcomer)
2. 📄 Specific Meaning in this Paper (How the authors use it)
3. 🧩 Intuitive Analogy
4. ⚙️ Why It Matters for this Research"""

                st.markdown(f"### 📚 Explanation: `{term}`")
                stream_gen = ask_groq_stream(
                    client=client,
                    model=selected_model,
                    system_prompt="You are PaperLens Word Explorer. Explain technical vocabulary clearly without losing scientific precision.",
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=1800,
                )
                st.write_stream(stream_gen)

            except Exception as e:
                st.error(f"Error explaining term: {e}")

    # --------------------------------------------------------------------------
    # FEATURE 5: Deep Dive (Granular Investigation)
    # --------------------------------------------------------------------------
    elif active_mode == "Deep Dive":
        st.header("🧪 Deep Dive")
        st.caption("Perform an exhaustive investigation of a specific subsection, algorithm, or theorem.")

        topic = st.text_input("Section, Method, or Experiment to Investigate:", placeholder="e.g., Section 4.2 Training Dynamics, Algorithm 1, Benchmark Ablations").strip()

        if st.button("🔬 Perform Deep Dive", type="primary", disabled=not topic):
            try:
                with st.spinner(f"Retrieving passages for '{topic}'..."):
                    retrieved = retriever.retrieve(topic, top_k=6)

                display_retrieval_citations(retrieved)

                context_str = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c, _ in retrieved)
                prompt = f"""Target Topic: {topic}

Retrieved Paper Passages:
{context_str}

Provide an exhaustive, in-depth breakdown:
1. Architectural / Conceptual Rationale
2. Step-by-Step Mathematical Formulation or Workflow
3. Critical Assumptions & Dependencies
4. Empirical Validation & Notable Findings
5. Shortcomings or Missing Details

Do not fabricate details not present in the excerpts. Cite page numbers whenever referencing specific claims."""

                st.markdown(f"### 🔬 Deep Dive: `{topic}`")
                stream_gen = ask_groq_stream(
                    client=client,
                    model=selected_model,
                    system_prompt="You are PaperLens Deep Dive. Provide a rigorous, academic analysis grounded in the provided text.",
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=3000,
                )
                st.write_stream(stream_gen)

            except Exception as e:
                st.error(f"Error performing deep dive: {e}")

    # --------------------------------------------------------------------------
    # 5.6 Diagnostics Drawer & Text Preview
    # --------------------------------------------------------------------------
    with st.expander("🛠️ Document Diagnostics & Full Text Viewer", expanded=False):
        st.markdown(f"**Filename:** `{paper['name']}`")
        st.markdown(f"**Extracted Title:** {paper['title']}")
        st.markdown(f"**Total Pages:** {paper['pages']} | **Readable Pages:** {paper['pages_with_text']}")
        st.markdown(f"**BM25 Indexed Chunks:** {len(paper['chunks'])}")

        st.text_area("Extracted Text Preview", value=paper["text"][:3000], height=220, disabled=True)
        st.download_button(
            "📥 Download Full Extracted Text (.txt)",
            data=paper["text"],
            file_name=f"{paper['name']}_extracted.txt",
            mime="text/plain",
        )


# Run main application
if __name__ == "__main__":
    main()