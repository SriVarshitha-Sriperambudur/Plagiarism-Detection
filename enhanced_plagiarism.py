import argparse
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import PyPDF2
from tqdm import tqdm

def read_txt(file_path):
    """Read content from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"Step 1: Reading file '{file_path}' - Successfully read {len(content)} characters.")
        return content
    except Exception as e:
        print(f"Step 1: Reading file '{file_path}' - Error: {e}")
        return None

def read_docx(file_path):
    """Read content from a .docx file."""
    try:
        doc = Document(file_path)
        content = ' '.join([para.text for para in doc.paragraphs if para.text.strip()])
        print(f"Step 1: Reading file '{file_path}' - Successfully read {len(content)} characters from .docx.")
        return content
    except Exception as e:
        print(f"Step 1: Reading file '{file_path}' - Error: {e}")
        return None

def read_pdf(file_path):
    """Read content from a .pdf file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                extracted = page.extract_text() or ''
                text += extracted
            print(f"Step 1: Reading file '{file_path}' - Successfully read {len(text)} characters from .pdf.")
            return text
    except Exception as e:
        print(f"Step 1: Reading file '{file_path}' - Error: {e}")
        return None

def read_file(file_path):
    """Read content from a file based on its extension."""
    print(f"\nNLP Process for file: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        return read_txt(file_path)
    elif ext == '.docx':
        return read_docx(file_path)
    elif ext == '.pdf':
        return read_pdf(file_path)
    else:
        print(f"Step 1: Reading file '{file_path}' - Error: Unsupported file format.")
        return None

def preprocess_text(text, file_path):
    """Preprocess text by converting to lowercase and removing extra whitespace."""
    if not text:
        print(f"Step 2: Preprocessing '{file_path}' - Error: No content to preprocess.")
        return ""
    original_len = len(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    print(f"Step 2: Preprocessing '{file_path}' - Converted to lowercase, removed extra whitespace. Original length: {original_len}, Processed length: {len(text)}.")
    return text

def calculate_similarity(texts, file_paths):
    """Calculate cosine similarity between texts using TF-IDF."""
    if not texts:
        print("Step 3: TF-IDF Vectorization - Error: No texts to process.")
        return []
    
    print("\nStep 3: TF-IDF Vectorization")
    print("-" * 40)
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        print(f"  - Created TF-IDF matrix with {tfidf_matrix.shape[0]} documents and {tfidf_matrix.shape[1]} unique terms.")
        print(f"  - Sample terms (first 5): {list(feature_names[:5])}")
        
        for i, file_path in enumerate(file_paths):
            terms = [(feature_names[j], tfidf_matrix[i,j]) for j in tfidf_matrix[i].nonzero()[1]]
            terms = sorted(terms, key=lambda x: x[1], reverse=True)[:5]
            print(f"  - Top 5 TF-IDF terms for '{os.path.basename(file_path)}': {terms}")
        
        print("\nStep 4: Cosine Similarity Calculation")
        print("-" * 40)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        print("  - Computed cosine similarity between all document pairs.")
        return similarity_matrix
    except Exception as e:
        print(f"Step 3 & 4: TF-IDF and Similarity - Error: {e}")
        return []

def generate_report(file_paths, similarity_matrix, output_dir="reports"):
    """Generate a plagiarism report and save it to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"plagiarism_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as report:
        report.write("Plagiarism Detection Report\n")
        report.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write("-" * 50 + "\n\n")
        report.write("Files Analyzed:\n")
        for i, file_path in enumerate(file_paths):
            report.write(f"{i+1}. {os.path.basename(file_path)}\n")
        report.write("\nSimilarity Results:\n")
        report.write("-" * 50 + "\n")
        for i in range(len(file_paths)):
            for j in range(i + 1, len(file_paths)):
                similarity = similarity_matrix[i][j] * 100
                report.write(f"Similarity between {os.path.basename(file_paths[i])} and {os.path.basename(file_paths[j])}: {similarity:.2f}%\n")
    
    print(f"\nStep 5: Reporting - Report saved to {report_path}")

def check_plagiarism(file_paths):
    """Compare all pairs of files and generate a report with NLP steps."""
    if len(file_paths) < 2:
        print("Error: At least two files are required for comparison.")
        return

    # Read and preprocess all files
    texts = []
    valid_files = []
    print("\nStarting NLP Plagiarism Detection Process")
    print("=" * 50)
    for file_path in tqdm(file_paths, desc="Processing files"):
        if not os.path.exists(file_path):
            print(f"Step 1: Reading file '{file_path}' - Error: File does not exist.")
            continue
        content = read_file(file_path)
        if content is not None:
            processed_text = preprocess_text(content, file_path)
            texts.append(processed_text)
            valid_files.append(file_path)

    if len(texts) < 2:
        print("Error: Not enough valid files to compare.")
        return

    # Calculate similarity
    similarity_matrix = calculate_similarity(texts, valid_files)

    # Print results
    print("\nPlagiarism Detection Results:")
    print("-" * 40)
    for i in range(len(valid_files)):
        for j in range(i + 1, len(valid_files)):
            similarity = similarity_matrix[i][j] * 100
            print(f"Similarity between {os.path.basename(valid_files[i])} and {os.path.basename(valid_files[j])}: {similarity:.2f}%")

    # Generate report
    generate_report(valid_files, similarity_matrix)

def main():
    parser = argparse.ArgumentParser(description="Plagiarism Detection Tool with NLP Steps")
    parser.add_argument('files', nargs='+', help="List of files to compare")
    args = parser.parse_args()

    check_plagiarism(args.files)

if __name__ == "__main__":
    main()