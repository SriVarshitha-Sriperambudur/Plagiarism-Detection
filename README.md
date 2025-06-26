# Plagiarism-Detection using Natural Language Processing

A powerful and extensible Plagiarism Detection Tool that uses Natural Language Processing (NLP) techniques like TF-IDF vectorization and cosine similarity to analyze and compare the textual similarity of documents (.txt, .docx, .pdf).

📌 Features:

✅ Supports .txt, .docx, and .pdf files

✅ Reads and preprocesses documents (case normalization, whitespace removal)

✅ Uses TF-IDF to extract top terms from each document

✅ Computes cosine similarity between each pair of files

✅ Generates a detailed plagiarism report with timestamp

✅ Handles missing files, unsupported formats, and errors gracefully

✅ Command-line interface (CLI) for batch comparison


🧠 NLP Techniques Used:

TF-IDF Vectorization: Transforms documents into numerical features while reducing the effect of common words.

Cosine Similarity: Measures the angle between vector representations to calculate document similarity.

Text Preprocessing: Lowercasing, whitespace cleanup, and token normalization.

