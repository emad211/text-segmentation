
```markdown
# Text Segmentation using GloVe and TF-IDF

## Description
This repository contains a Python project for segmenting text documents using GloVe embeddings and TF-IDF. The project reads text files, tokenizes them, calculates word statistics, and segments the text into meaningful parts. The segments are then saved into output files.

## Features
- Tokenizes and processes text documents.
- Calculates word statistics using TF-IDF.
- Segments text using GloVe embeddings.
- Saves segmented texts to output files.

## Requirements
- Python 3.x
- spaCy
- nltk
- gensim
- numpy
- scikit-learn
- scipy
- matplotlib

## Installation
To install the required dependencies, run:
```bash
pip install spacy nltk gensim numpy scikit-learn scipy matplotlib
python -m spacy download en_core_web_sm
```

## Usage
To use the text segmentation script, follow these steps:

1. Place your text files in a directory.
2. Ensure you have the GloVe embeddings file (`glove.6B.50d.word2vec.txt`).
3. Run the script to segment the text files and save the segments.

Example usage:
```bash
python import_spacy.py
```

## Example
Here is an example of how to set up and run the script:
```python
# Directory containing your text files
input_dir = "path/to/your/text/files"
output_dir = "path/to/save/segmented/files"
glove_file = "glove.6B.50d.word2vec.txt"  # Ensure this is the correct path to your GloVe file

# Process each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):  # Only process text files
        file_path = os.path.join(input_dir, filename)
        process_and_save_file(file_path, glove_file, output_dir)
```

## Script Details
### Files:
- **`import_spacy.py`**: Main script for reading, processing, and segmenting text files.

### Classes and Functions:
- **`Text`**: Represents a text document.
- **`Segment`**: Represents a segment of the text.
- **`text_segmentation_class`**: Main class for text segmentation.
- **`save_segments_to_file(text_object, output_file)`**: Saves the segments to a file.
- **`process_and_save_file(file_path, glove_file, output_dir)`**: Processes a text file and saves the segments.


## Contact
For any questions or suggestions, feel free to contact me at emad.k5000@gmail.com
```

