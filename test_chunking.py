
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample research paper text for testing
paper_text = """
Artificial Intelligence in Healthcare

Introduction
Artificial Intelligence is transforming healthcare by helping
doctors analyse medical data and detect diseases earlier.
Machine learning models can identify patterns in medical images.

Methodology
The researchers trained a deep learning model using labelled
medical images. The dataset was divided into training,
validation, and testing subsets. The model was evaluated
using accuracy, precision, recall, and F1-score.

Results
The model achieved promising results on the test dataset.
However, its performance depended on the quality and diversity
of the training data.

Conclusion
AI can support clinical decision-making, but further validation
is needed before deployment in real-world healthcare settings.
"""

# Configure the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Split the text into chunks
chunks = splitter.split_text(paper_text)

# Display the results
print(f"Total chunks created: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)
    print(f"Characters: {len(chunk)}")
