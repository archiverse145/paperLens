
import streamlit as st
from pypdf import PdfReader
from io import BytesIO

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PaperLens",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Application header
# -----------------------------
st.title("PaperLens")
st.subheader("Your AI Research Paper Assistant")

st.write(
    "Upload a research paper to begin understanding its content. "
    "PaperLens will help you explore the document before we add "
    "AI-powered summaries and question answering."
)

st.divider()

# -----------------------------
# PDF upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your research paper",
    type=["pdf"],
    accept_multiple_files=False,
    max_upload_size=20
)

if uploaded_file is not None:
    try:
        # Read the uploaded PDF
        pdf_bytes = uploaded_file.getvalue()
        reader = PdfReader(BytesIO(pdf_bytes))

        # Check whether the PDF is encrypted
        if reader.is_encrypted:
            st.error(
                "This PDF is encrypted. Please upload an "
                "unencrypted PDF that can be read."
            )
            st.stop()

        # Extract text from each page
        extracted_pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            extracted_pages.append(page_text)

        full_text = "\n\n".join(extracted_pages).strip()

        # Read document metadata when available
        metadata = reader.metadata
        title = metadata.title if metadata else None

        if not title:
            title = uploaded_file.name

        # -----------------------------
        # Display document information
        # -----------------------------
        st.success("PDF uploaded and processed successfully.")
        st.header(title)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total pages", len(reader.pages))

        with col2:
            st.metric("Characters extracted", len(full_text))

        with col3:
            st.metric(
                "Pages with text",
                sum(bool(text.strip()) for text in extracted_pages)
            )

        st.divider()

        # -----------------------------
        # Handle empty or scanned PDFs
        # -----------------------------
        if not full_text:
            st.warning(
                "No extractable text was found. This may be a "
                "scanned PDF. OCR support can be added later."
            )

        else:
            st.subheader("Extracted paper text")

            preview_length = 5000

            with st.expander("View text preview", expanded=True):
                st.text_area(
                    "Extracted content",
                    value=full_text[:preview_length],
                    height=350,
                    disabled=True
                )

                if len(full_text) > preview_length:
                    st.caption(
                        "Showing the first 5,000 characters. "
                        "The complete extracted text is available "
                        "for download below."
                    )

            # Download extracted text
            st.download_button(
                label="Download extracted text",
                data=full_text,
                file_name="paperlens_extracted_text.txt",
                mime="text/plain"
            )

    except Exception as error:
        st.error(
            "PaperLens could not process this PDF. "
            "Try another PDF or check whether the file is valid."
        )

        with st.expander("Technical details"):
            st.code(str(error))

else:
    st.info("Upload a PDF above to get started.")

    st.markdown(
        """
        **What we will build next**
        - Structured research paper summaries
        - Prerequisites explained in simple language
        - Section-by-section explanations
        - Difficult-word definitions
        - Question answering grounded in the paper
        """
    )
