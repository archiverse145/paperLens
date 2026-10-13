
"""
PaperLens: a Streamlit research-paper assistant.

Features:
- Understand Paper: guided, first-read report.
- Paper Summary: concise revision notes.
- Ask the Paper: answers grounded in relevant paper passages.
- Word Explorer: explain a term in the paper's context.
- Deep Dive: investigate a section, method, or experiment.

Setup:
1. Install the required Python packages.
2. Put GROQ_API_KEY=your_key_here in a .env file beside app.py.
3. Run: streamlit run app.py
"""

# -----------------------------
# Imports
# -----------------------------
import os
from io import BytesIO
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# -----------------------------
# Page configuration and setup
# -----------------------------
st.set_page_config(
    page_title="PaperLens",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment variables from the .env file.
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API key not found. Add GROQ_API_KEY=... to your .env file.")
    st.stop()

# Create the Groq client.
client = Groq(api_key=GROQ_API_KEY)

# Model used by all AI features.
MODEL_NAME = "openai/gpt-oss-20b"


# -----------------------------
# Shared AI helper
# -----------------------------
def ask_groq(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1800
) -> str:
    """Send one request to Groq and return the generated text."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError(
            "The model returned an empty response. Please try again."
        )

    return answer.strip()


# -----------------------------
# Find relevant paper excerpts
# -----------------------------
def get_relevant_chunks(
    paper: Dict,
    query: str,
    limit: int = 4
) -> List[str]:
    """
    Select text chunks with the most query-term overlap.

    This is a lightweight keyword search. It does not use embeddings,
    so it can miss passages that use different wording.
    """

    chunks = paper.get("chunks", [])

    if not chunks:
        return []

    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "what",
        "when", "where", "which", "does", "into", "about", "are", "was",
        "were", "how", "why", "can", "could", "would", "should", "paper",
        "explain", "please", "tell", "describe", "give", "its", "their",
        "have", "has", "had", "not", "but", "you", "your", "use", "used",
    }

    cleaned_query = query.lower()

    for char in ".,?!:;()[]{}\"'`/\\":
        cleaned_query = cleaned_query.replace(char, " ")

    terms = {
        word
        for word in cleaned_query.split()
        if len(word) >= 3 and word not in stop_words
    }

    # If the query has no useful terms, use the first chunks.
    if not terms:
        return chunks[:limit]

    scored: List[Tuple[int, int, str]] = []

    for index, chunk in enumerate(chunks):
        lower_chunk = chunk.lower()
        score = sum(lower_chunk.count(term) for term in terms)

        if score:
            scored.append((score, -index, chunk))

    # Sort by score, keeping earlier chunks ahead when scores tie.
    scored.sort(reverse=True)

    selected = [item[2] for item in scored[:limit]]

    # Fall back to the first chunks if no terms match.
    if selected:
        return selected

    return chunks[:min(2, len(chunks))]


# -----------------------------
# PDF processing
# -----------------------------
def process_pdf(uploaded_file) -> Dict:
    """Extract text, metadata, and searchable chunks from a PDF."""

    reader = PdfReader(BytesIO(uploaded_file.getvalue()))

    if reader.is_encrypted:
        raise ValueError(
            "This PDF is encrypted. Upload an unencrypted PDF."
        )

    # Extract text from each page.
    extracted_pages = [
        (page.extract_text() or "").strip()
        for page in reader.pages
    ]

    # Join non-empty pages into one document.
    full_text = "\n\n".join(
        text for text in extracted_pages if text
    ).strip()

    chunks: List[str] = []

    # Split extracted text into manageable chunks.
    if full_text:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_text(full_text)

    # Use the PDF metadata title when available.
    metadata = reader.metadata
    title = metadata.title.strip() if metadata and metadata.title else ""

    if not title:
        title = uploaded_file.name

    return {
        "name": uploaded_file.name,
        "title": title,
        "pages": len(reader.pages),
        "text": full_text,
        "chunks": chunks,
        "pages_with_text": sum(
            bool(text) for text in extracted_pages
        ),
        "processed": True,
    }


# -----------------------------
# Understand Paper: report generation
# -----------------------------
def generate_understanding_report(paper_text: str) -> str:
    """Generate a detailed first-read report using a two-pass process."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=7000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(paper_text)

    if not chunks:
        raise ValueError("No readable text was found in the paper.")

    section_notes = []

    # First pass: analyse each part of the paper.
    for index, chunk in enumerate(chunks, start=1):
        notes = ask_groq(
            (
                "You are PaperLens, a precise research-paper tutor. "
                "Never invent facts absent from the supplied text."
            ),
            f"""Analyze this part of a research paper.

Identify, when present:
- Section headings
- Main concepts and arguments
- Methodology and experiment steps
- Findings and limitations
- Technical terms that need explanation

Preserve technical details. Do not assume a finding or experiment
exists if it is not mentioned.

PAPER SEGMENT {index} OF {len(chunks)}:
{chunk}""",
            max_tokens=1800,
        )

        section_notes.append(notes)

    notes_text = "\n\n".join(section_notes)

    # Second pass: combine notes into a guided reading report.
    return ask_groq(
        (
            "You are PaperLens, an expert research-paper tutor. "
            "Produce an accurate, structured teaching report."
        ),
        f"""Create a guided reading report from the research notes below.

Use these sections where supported by the paper:
1. Prerequisites
2. Research Problem and Motivation
3. Section-by-Section Explanation
4. Methodology and Experiments
5. Key Findings and Results
6. Limitations
7. Technical Details

Requirements:
- Follow the paper's actual headings and order wherever identifiable.
- Explain concepts simply without losing technical accuracy.
- Explain experiment steps only when supported by the notes.
- Put advanced technical details near the end.
- Do not invent findings, headings, or experimental steps.
- If information is unavailable, say so.
- Do not mention page numbers.
- Avoid tables unless needed to explain an experiment.

RESEARCH NOTES:
{notes_text}""",
        max_tokens=6000,
    )


# -----------------------------
# Paper Summary: revision notes
# -----------------------------
def generate_paper_summary(paper_text: str) -> str:
    """Generate a concise, structured summary for later revision."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(paper_text)

    if not chunks:
        raise ValueError("No readable text was found in the paper.")

    chunk_notes = []

    # First pass: summarise each part.
    for index, chunk in enumerate(chunks, start=1):
        notes = ask_groq(
            (
                "You create accurate, concise research-paper revision "
                "notes. Never invent information."
            ),
            f"""Summarize this part of a research paper for revision.

Preserve identifiable section headings. Capture the main argument,
method, important technical details, findings, and limitations when
present. Keep the notes concise but meaningful. Do not add facts
unsupported by the text.

PART {index} OF {len(chunks)}:
{chunk}""",
            max_tokens=1400,
        )

        chunk_notes.append(notes)

    notes_text = "\n\n".join(chunk_notes)

    # Second pass: create the final revision-friendly summary.
    return ask_groq(
        "You are PaperLens, an accurate academic summarization assistant.",
        f"""Create a revision-friendly summary from these research notes.

Requirements:
- Follow the paper's original headings and order wherever identifiable.
- Do not force unrelated content into invented sections.
- Explain the main idea of each section concisely.
- Include methodology, key results, and limitations when present.
- Retain essential technical details and terminology.
- Do not mention page numbers.
- Do not invent findings or claims.
- Prefer headings and bullet points over tables.

RESEARCH NOTES:
{notes_text}""",
        max_tokens=5000,
    )

# -----------------------------
# Session state
# -----------------------------
# Store uploaded paper data and generated answers across Streamlit reruns.
if "paper_data" not in st.session_state:
    st.session_state.paper_data = None

if "active_feature" not in st.session_state:
    st.session_state.active_feature = "Understand Paper"

# Each feature stores its own result.
for state_key in (
    "understanding_report",
    "paper_summary",
    "paper_answers",
    "word_explanation",
    "deep_dive_result",
):
    if state_key not in st.session_state:
        st.session_state[state_key] = None


# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(91, 100, 240, 0.16),
            rgba(36, 180, 170, 0.10)
        );
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.5rem;
    }

    .hero-label {
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.75;
    }

    .feature-card {
        padding: 1.2rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        min-height: 145px;
        margin-bottom: 0.8rem;
    }

    .muted {
        opacity: 0.7;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("# 📄 PaperLens")
    st.caption("YOUR RESEARCH COMPANION")
    st.divider()

    st.markdown("**WORKSPACE**")

    # The radio selection determines which feature appears in the main area.
    features = {
        "Understand Paper": "🧠",
        "Paper Summary": "📝",
        "Ask the Paper": "💬",
        "Word Explorer": "🔎",
        "Deep Dive": "🧪",
    }

    st.radio(
        "WORKSPACE",
        options=list(features.keys()),
        format_func=lambda feature: f"{features[feature]}  {feature}",
        key="active_feature",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**CURRENT PAPER**")

    paper = st.session_state.paper_data

    if paper:
        st.markdown(f"📄 **{paper['name']}**")
        st.caption(
            f"{paper['pages']} pages · "
            f"{len(paper['chunks'])} text chunks"
        )

        # Removing a paper also clears generated results and widget values.
        if st.button("Remove current paper", use_container_width=True):
            st.session_state.paper_data = None

            for state_key in (
                "understanding_report",
                "paper_summary",
                "paper_answers",
                "word_explanation",
                "deep_dive_result",
            ):
                st.session_state[state_key] = None

            for widget_key in (
                "paper_question",
                "word_to_explore",
                "deep_dive_topic",
            ):
                if widget_key in st.session_state:
                    del st.session_state[widget_key]

            st.rerun()
    else:
        st.caption("No paper uploaded yet.")

    st.divider()

    # Uploading a new PDF replaces the current paper after processing succeeds.
    with st.expander("Upload or replace paper"):
        new_file = st.file_uploader(
            "Choose a research paper",
            type=["pdf"],
            accept_multiple_files=False,
            max_upload_size=20,
            key="paper_uploader",
        )

        if st.button(
            "Process PDF",
            use_container_width=True,
            disabled=new_file is None,
        ):
            try:
                with st.spinner("Reading your research paper..."):
                    new_paper = process_pdf(new_file)

                # Do not retain results generated for the previous document.
                st.session_state.paper_data = new_paper

                for state_key in (
                    "understanding_report",
                    "paper_summary",
                    "paper_answers",
                    "word_explanation",
                    "deep_dive_result",
                ):
                    st.session_state[state_key] = None

                for widget_key in (
                    "paper_question",
                    "word_to_explore",
                    "deep_dive_topic",
                ):
                    if widget_key in st.session_state:
                        del st.session_state[widget_key]

                st.success("Paper processed successfully.")
                st.rerun()

            except Exception as error:
                st.error(f"Could not process this PDF: {error}")

# -----------------------------
# Main workspace
# -----------------------------
paper = st.session_state.paper_data
active = st.session_state.active_feature

# Display the PaperLens hero banner.
st.markdown(
    """
    <div class="hero">
        <div class="hero-label">READ · UNDERSTAND · REVISE</div>
        <h1>PaperLens</h1>
        <p>Your research paper, made easier to understand.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Empty state: no paper uploaded
# -----------------------------
if paper is None:
    st.subheader("Your research workspace")
    st.write(
        "Upload one research paper using the sidebar to get started. "
        "Your workspace will then be ready for paper-specific analysis."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🧠 Understand Paper</h3>
                <p>Learn the background, prerequisites, and ideas
                behind the research.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>💬 Ask the Paper</h3>
                <p>Ask questions and get answers grounded in
                the uploaded document.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>🧪 Deep Dive</h3>
                <p>Explore a particular section, methodology,
                experiment, or result.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📝 Paper Summary</h3>
                <p>Generate a structured, revision-friendly
                summary that follows the paper's headings.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>🔎 Word Explorer</h3>
                <p>Understand difficult words and technical
                terms in context.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Choose a feature from the sidebar at any time. "
        "Upload your paper there to begin."
    )


# -----------------------------
# Workspace: paper uploaded
# -----------------------------
else:
    st.caption("YOUR CURRENT PAPER")
    st.title(paper["title"])

    # Show basic information about the uploaded document.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pages", paper["pages"])

    with col2:
        st.metric(
            "Characters extracted",
            f"{len(paper['text']):,}",
        )

    with col3:
        st.metric("Text chunks", len(paper["chunks"]))

    if not paper["text"]:
        st.warning(
            "No extractable text was found. This PDF may be scanned. "
            "OCR support is not included in this version."
        )

    st.divider()

    # -----------------------------
    # Feature 1: Understand Paper
    # -----------------------------
    if active == "Understand Paper":
        st.header("Understand Paper")
        st.write(
            "Build your understanding before diving into the details."
        )

        st.markdown(
            """
            Your guided reading report will cover:

            - Prerequisites explained simply
            - The paper's actual sections and main ideas
            - Methodology and experiments, where applicable
            - Key findings, limitations, and technical details
            """
        )

        if not paper["text"]:
            st.warning(
                "This PDF contains no extractable text, "
                "so PaperLens cannot analyse it yet."
            )

        elif st.button(
            "Generate understanding report",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Reading your paper and generating explanations..."
                ):
                    report = generate_understanding_report(paper["text"])

                st.session_state["understanding_report"] = {
                    "paper_name": paper["name"],
                    "content": report,
                }

            except Exception as error:
                st.error("PaperLens could not generate the report.")

                with st.expander("Technical details"):
                    st.code(str(error))
        
        # Display the saved report only for the current paper.
        saved_report = st.session_state.get("understanding_report")

        if (
            saved_report
            and saved_report.get("paper_name") == paper["name"]
        ):
            st.divider()
            st.subheader("Your guided reading report")
            st.markdown(saved_report["content"])

            st.download_button(
                "Download understanding report",
                data=saved_report["content"],
                file_name="paperlens_understanding_report.md",
                mime="text/markdown",
            )


    # -----------------------------
    # Feature 2: Paper Summary
    # -----------------------------
    elif active == "Paper Summary":
        st.header("Paper Summary")
        st.write(
            "Create a structured summary designed for quick revision."
        )

        st.markdown(
            """
            The summary follows the paper's identifiable headings,
            explains the main ideas, and retains important technical details.
            """
        )

        if not paper["text"]:
            st.warning(
                "No extractable text was found, so a summary "
                "cannot be generated."
            )

        elif st.button(
            "Generate paper summary",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Generating your revision summary..."):
                    summary = generate_paper_summary(paper["text"])

                st.session_state["paper_summary"] = {
                    "paper_name": paper["name"],
                    "content": summary,
                }

            except Exception as error:
                st.error("PaperLens could not generate the summary.")

                with st.expander("Technical details"):
                    st.code(str(error))

        # Display the summary for the current paper only.
        saved_summary = st.session_state.get("paper_summary")

        if (
            saved_summary
            and saved_summary.get("paper_name") == paper["name"]
        ):
            st.divider()
            st.subheader("Your revision summary")
            st.markdown(saved_summary["content"])

            st.download_button(
                "Download paper summary",
                data=saved_summary["content"],
                file_name="paperlens_summary.md",
                mime="text/markdown",
            )

    # -----------------------------
    # Feature 3: Ask the Paper
    # -----------------------------
    elif active == "Ask the Paper":
        st.header("Ask the Paper")
        st.write(
            "Ask questions about the uploaded research paper."
        )

        question = st.text_area(
            "Your question",
            placeholder="What problem is this paper trying to solve?",
            key="paper_question",
        ).strip()

        if not paper["text"]:
            st.warning(
                "No extractable text is available for question answering."
            )

        elif st.button(
            "Ask question",
            type="primary",
            disabled=not question,
        ):
            try:
                with st.spinner("Finding the answer in your paper..."):
                    relevant_chunks = get_relevant_chunks(
                        paper,
                        question,
                        limit=4,
                    )

                    if not relevant_chunks:
                        raise ValueError(
                            "No readable text chunks are available."
                        )

                    context = "\n\n".join(relevant_chunks)

                    answer = ask_groq(
                        (
                            "You are PaperLens, a research paper assistant. "
                            "Answer using the supplied paper excerpts. "
                            "Explain concepts clearly and do not invent facts. "
                            "If the excerpts do not contain enough information, "
                            "state that limitation explicitly."
                        ),
                        f"""Question:
{question}

Relevant paper excerpts:
{context}

Answer the question clearly. Distinguish paper-supported facts
from any general explanation that may help the reader.""",
                        max_tokens=1800,
                    )

                st.session_state["paper_answers"] = {
                    "paper_name": paper["name"],
                    "question": question,
                    "answer": answer,
                }

            except Exception as error:
                st.error("PaperLens could not answer the question.")

                with st.expander("Technical details"):
                    st.code(str(error))

        # Show the latest answer for this document.
        saved_answer = st.session_state.get("paper_answers")

        if (
            saved_answer
            and saved_answer.get("paper_name") == paper["name"]
        ):
            st.markdown("### Answer")
            st.markdown(saved_answer["answer"])

        st.caption(
            "Answers use selected excerpts from the paper. "
            "Always check the original text for critical details."
        )

    # -----------------------------
    # Feature 4: Word Explorer
    # -----------------------------
    elif active == "Word Explorer":
        st.header("Word Explorer")
        st.write(
            "Understand unfamiliar words, abbreviations, and technical terms."
        )

        term = st.text_input(
            "Word or technical term",
            placeholder="e.g., transformer, attention mechanism, ablation",
            key="word_to_explore",
        ).strip()

        if not paper["text"]:
            st.warning(
                "No extractable text is available for Word Explorer."
            )

        elif st.button(
            "Explain term",
            type="primary",
            disabled=not term,
        ):
            try:
                with st.spinner("Explaining the term in context..."):
                    relevant_chunks = get_relevant_chunks(
                        paper,
                        term,
                        limit=4,
                    )

                    context = "\n\n".join(relevant_chunks)

                    explanation = ask_groq(
                        (
                            "You are PaperLens Word Explorer. Explain "
                            "technical terms clearly and accurately. "
                            "Distinguish general definitions from claims "
                            "supported by the supplied paper excerpts. "
                            "Never invent paper-specific information."
                        ),
                        f"""Explain this term for a reader studying a research paper.

TERM:
{term}

PAPER EXCERPTS:
{context}

Organize your answer using these headings:
1. Simple Meaning
2. Meaning in This Paper
3. Example or Analogy
4. Why It Matters

If the excerpts do not mention the term directly, say so and provide
a general explanation only where appropriate. Do not pretend that
the paper supports an explanation that is not present in the excerpts.""",
                        max_tokens=1400,
                    )

                st.session_state["word_explanation"] = {
                    "paper_name": paper["name"],
                    "term": term,
                    "answer": explanation,
                }

            except Exception as error:
                st.error("PaperLens could not explain this term.")

                with st.expander("Technical details"):
                    st.code(str(error))

        saved_word = st.session_state.get("word_explanation")

        if (
            saved_word
            and saved_word.get("paper_name") == paper["name"]
        ):
            st.divider()
            st.subheader(f"Explanation: {saved_word['term']}")
            st.markdown(saved_word["answer"])

    # -----------------------------
    # Feature 5: Deep Dive
    # -----------------------------
    elif active == "Deep Dive":
        st.header("Deep Dive")
        st.write(
            "Investigate a specific section, methodology, "
            "equation, or experiment."
        )

        topic = st.text_input(
            "Section or topic to explore",
            placeholder="e.g., Proposed Method, Experimental Setup",
            key="deep_dive_topic",
        ).strip()

        if not paper["text"]:
            st.warning(
                "No extractable text is available for Deep Dive."
            )

        elif st.button(
            "Explore section",
            type="primary",
            disabled=not topic,
        ):
            try:
                with st.spinner("Analysing the relevant paper excerpts..."):
                    relevant_chunks = get_relevant_chunks(
                        paper,
                        topic,
                        limit=5,
                    )

                    context = "\n\n".join(relevant_chunks)

                    analysis = ask_groq(
                        (
                            "You are PaperLens Deep Dive, a research-paper "
                            "analysis assistant. Stay grounded in the "
                            "provided excerpts. Never invent experimental "
                            "steps, equations, results, or claims."
                        ),
                        f"""Explain this research-paper topic in depth.

TOPIC:
{topic}

PAPER EXCERPTS:
{context}

Use the following structure where relevant and supported:
1. Overview and Purpose
2. Core Idea
3. How It Works
4. Step-by-Step Method or Experiment
5. Important Technical Concepts
6. Results and Significance
7. Limitations or Missing Information

Explain difficult concepts clearly while preserving technical accuracy.
For an experiment, explain its procedure step by step only when
the excerpts provide enough detail. If a requested topic is not
covered by the excerpts, clearly state that limitation.""",
                        max_tokens=2400,
                    )

                st.session_state["deep_dive_result"] = {
                    "paper_name": paper["name"],
                    "topic": topic,
                    "answer": analysis,
                }

            except Exception as error:
                st.error("PaperLens could not analyse this topic.")

                with st.expander("Technical details"):
                    st.code(str(error))

        saved_dive = st.session_state.get("deep_dive_result")

        if (
            saved_dive
            and saved_dive.get("paper_name") == paper["name"]
        ):
            st.divider()
            st.subheader(f"Deep Dive: {saved_dive['topic']}")
            st.markdown(saved_dive["answer"])

            st.download_button(
                "Download Deep Dive",
                data=saved_dive["answer"],
                file_name="paperlens_deep_dive.md",
                mime="text/markdown",
            )


# -----------------------------
# Optional document diagnostics
# -----------------------------
# Keep this section OUTSIDE the feature conditions.
# It must also remain separate from the feature_descriptions dictionary.

with st.expander("Document diagnostics"):
    current_paper = st.session_state.paper_data

    if current_paper is None:
        st.info(
            "Upload and process a research paper to view diagnostics."
        )

    else:
        st.write(f"**File:** {current_paper['name']}")
        st.write(f"**Pages:** {current_paper['pages']}")
        st.write(
            "**Pages containing text:** "
            f"{current_paper['pages_with_text']}"
        )
        st.write(
            f"**Total text chunks:** {len(current_paper['chunks'])}"
        )
        st.write(
            f"**Extracted characters:** {len(current_paper['text']):,}"
        )

        if current_paper["text"]:
            st.text_area(
                "Extracted text preview",
                value=current_paper["text"][:5000],
                height=250,
                disabled=True,
                key="diagnostic_preview",
            )

            st.download_button(
                "Download extracted text",
                data=current_paper["text"],
                file_name="paperlens_extracted_text.txt",
                mime="text/plain",
            )

        else:
            st.warning(
                "No extractable text was found. "
                "Scanned PDFs require OCR, which is not included here."
            )


# -----------------------------
# Feature descriptions
# -----------------------------
feature_descriptions = {
    "Understand Paper": (
        "Your first-read companion. Learn the prerequisites, "
        "understand major sections, follow the proposed approach, "
        "and explore experiments in simple language."
    ),
    "Paper Summary": (
        "Revision-ready notes following the paper's identifiable "
        "headings, with key contributions, results, limitations, "
        "and technical takeaways."
    ),
    "Ask the Paper": (
        "Ask questions in natural language and receive answers "
        "based on selected excerpts from the uploaded paper."
    ),
    "Word Explorer": (
        "Explore unfamiliar words, abbreviations, and technical "
        "terms with definitions and paper-specific context."
    ),
    "Deep Dive": (
        "Investigate a heading, method, equation, experiment, "
        "result, or limitation in greater technical depth."
    ),
}
