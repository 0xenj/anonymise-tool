import re
from flask import Flask, request, render_template, send_from_directory
import os
import shutil
import zipfile
import docx2txt
from PyPDF2 import PdfFileReader
import easyocr
from PIL import Image
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_cors import CORS
import logging
import time

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Helper function to load the NLP pipeline
def get_nlp_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp

# Helper function to load EasyOCR reader
def get_easyocr_reader():
    return easyocr.Reader(['fr'], gpu=False)

# Text cleaning functions
def removeCharacters(x):
    # Remove specific characters from text
    x = x.str.replace(r'\'text\'', '')
    x = x.str.replace(r'\'start\'', '')
    x = x.str.replace(r'\'end\'', '')
    x = x.str.replace(r'[{}<>]', '')
    x = x.str.replace(r'[]', '')
    x = x.str.replace(r'["]', '')
    x = x.str.replace(r'[\']', '')
    x = x.str.replace(r'[:]', '')
    x = x.str.replace(r'[()]', '')
    x = x.str.replace(r'[?]', '')
    return x

def extract_report(text):
    pattern1 = r"(?i)(indications?|indication) ?\s*:? ?(.*?)(?=\bdocteur\b)"
    pattern2 = r"(?i)(.*?)(indications?|indication)"
    match1 = re.search(pattern1, text, re.DOTALL)
    match2 = re.search(pattern2, text, re.DOTALL)
    titre = ''
    if match2:
        before_indication = match2.group(1).strip()
        sentences = re.split(r'(?<=[.!?])\s+', before_indication)
        titre = (sentences[-1].split('\n')[-1])
    if match1:
        return titre.lower() + '\n' + (match1.group(2).strip())
    else:
        return text

def remove_technique_section(text):
    pattern = r'(?i)\b(?:technique|techniques|technique:).*?(?=\b(?:résultat|resultat|résultats|resultats|résultats:|resultats:)\b)'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned_text

# Function to anonymize text
def remove_person_entities(text):
    nlp = get_nlp_pipeline()
    entities = nlp(text)
    person_entities = [entity for entity in entities if entity['entity_group'] == 'PER']
    for entity in sorted(person_entities, key=lambda x: x['start'], reverse=True):
        text = text[:entity['start']] + ' ... ' + text[entity['end']:]
    return text

# Function to extract text from PDF files
def read_pdf(file_path):
    pdfReader = PdfFileReader(file_path)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()
    return all_page_text

# Function to extract text from DOCX files
def readtext_word_file(file_path):
    return docx2txt.process(file_path)

# Function to extract text from images using EasyOCR
def extract_text_with_easyocr(file_path):
    reader = get_easyocr_reader()
    image = Image.open(file_path)
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0)
    return " ".join(result)

# Function to process each file, extract text, anonymize it, and save as .txt
def process_file(file_path, output_path):
    extension = file_path.rsplit('.', 1)[1].lower()
    if extension == 'docx':
        text = readtext_word_file(file_path)
    elif extension == 'pdf':
        text = read_pdf(file_path)
    elif extension in ['png', 'jpg', 'jpeg']:
        text = extract_text_with_easyocr(file_path)
    else:
        return

    text = extract_report(text)
    text = remove_technique_section(text)
    anonymized_text = remove_person_entities(text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(anonymized_text)

# Function to process all files in the folder
def process_files_in_folder(upload_folder, processed_folder):
    for root, dirs, files in os.walk(upload_folder):
        for file in files:
            file_path = os.path.join(root, file)
            output_file = os.path.join(processed_folder, f"{os.path.splitext(file)[0]}.txt")
            process_file(file_path, output_file)

# Flask routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def upload_files():
    logging.info(f"Upload route accessed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    shutil.rmtree(UPLOAD_FOLDER)
    shutil.rmtree(PROCESSED_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    files = request.files.getlist('file')
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        logging.info(f"File saved and confirmed at {file_path}")

        if not os.path.exists(file_path):
            app.logger.error(f"File {file_path} not found after saving.")
            return "File not found", 500
        else:
            app.logger.info(f"File confirmed saved at {file_path}")
        
        time.sleep(0.1)

    logging.info(f"Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    process_files_in_folder(UPLOAD_FOLDER, PROCESSED_FOLDER)

    processed_files = os.listdir(PROCESSED_FOLDER)
    if not processed_files:
        logging.error("No files found in the processed folder after processing.")
        return "Processing failed. No files in output.", 500

    zip_filename = "processed_files.zip"
    zip_filepath = os.path.join(PROCESSED_FOLDER, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for root, dirs, files in os.walk(PROCESSED_FOLDER):
            for file in files:
                if file != zip_filename:
                    zipf.write(os.path.join(root, file), arcname=file)

    logging.info(f"Zip file created at {zip_filepath}")
    return send_from_directory(PROCESSED_FOLDER, zip_filename, as_attachment=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return "Le fichier est trop volumineux ! La taille maximale autorisée est de {} Mo.".format(app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)), 413

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
