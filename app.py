import re  # Pour le script que tu as fourni
from flask import Flask, request, render_template, send_from_directory
import os
import shutil
import zipfile
import docx2txt
from PyPDF2 import PdfFileReader
import easyocr
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_cors import CORS

# Initialize the Huggingface model for anonymization
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Initialize the easyocr reader
reader = easyocr.Reader(['fr'], gpu=False)  # Set gpu=True if you have GPU support

app = Flask(__name__)

CORS(app)

# Dossiers pour l'upload et les fichiers traités
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# --- Nouvelles fonctions de traitement du texte ---
def removeCharacters(x):
    # Retirer les caractères spécifiques du texte
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
    # Extraire les sections d'indication du rapport médical
    pattern1 = r"(?i)(indications?|indication) ?\s*:? ?(.*?)(?=\bdocteur\b)"
    pattern2 = r"(?i)(.*?)(indications?|indication)"
    
    match1 = re.search(pattern1, text, re.DOTALL)
    match2 = re.search(pattern2, text, re.DOTALL)
    titre = ''
    
    if match2:
        before_indication = match2.group(1).strip()
        sentences = re.split(r'(?<=[.!?])\s+', before_indication)
        titre = (sentences[-1].split('\n')[-1])  # Dernière phrase avant l'indication

    if match1:
        return titre.lower() + '\n' + (match1.group(2).strip())
    else:
        return text

def remove_technique_section(text):
    # Retirer la section technique du texte
    pattern = r'(?i)\b(?:technique|techniques|technique:).*?(?=\b(?:résultat|resultat|résultats|resultats|résultats:|resultats:)\b)'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned_text

# --------------------------------------------------

# Anonymization function to remove person entities
def remove_person_entities(text):
    entities = nlp(text)
    person_entities = [entity for entity in entities if entity['entity_group'] == 'PER']
    for entity in sorted(person_entities, key=lambda x: x['start'], reverse=True):
        text = text[:entity['start']] + ' ... ' + text[entity['end']:]  # Replace the entity with "..."
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

# Function to extract text from images using easyocr
def extract_text_with_easyocr(file_path):
    image = Image.open(file_path)
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0)
    return " ".join(result)

# Function to process each file, extract text, anonymize it, and save as .txt
def process_file(file_path, output_path):
    extension = file_path.rsplit('.', 1)[1].lower()

    if extension == 'docx':
        # Extraction du texte d'un fichier Word
        text = readtext_word_file(file_path)
    elif extension == 'pdf':
        # Extraction du texte d'un fichier PDF
        text = read_pdf(file_path)
    elif extension in ['png', 'jpg', 'jpeg']:  # Si le fichier est une image, utiliser OCR
        text = extract_text_with_easyocr(file_path)
    else:
        return  # Fichier non supporté

    # --- Appliquer les étapes supplémentaires de parsing ---
    text = extract_report(text)  # Extraction de la partie "Indications"
    text = remove_technique_section(text)  # Suppression de la section technique

    # Anonymiser le texte
    anonymized_text = remove_person_entities(text)

    # Sauvegarder dans un fichier texte
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(anonymized_text)

# Function to process all files in the folder
def process_files_in_folder(upload_folder, processed_folder):
    for root, dirs, files in os.walk(upload_folder):
        for file in files:
            file_path = os.path.join(root, file)
            output_file = os.path.join(processed_folder, f"{os.path.splitext(file)[0]}.txt")
            process_file(file_path, output_file)

# Page d'accueil (GET)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route pour uploader et traiter les fichiers (POST)
@app.route('/uploads', methods=['POST'])
def upload_files():
    # Réinitialiser les dossiers upload et processed
    shutil.rmtree(UPLOAD_FOLDER)
    shutil.rmtree(PROCESSED_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    # Sauvegarder tous les fichiers uploadés
    files = request.files.getlist('file')
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

    # Traiter les fichiers
    process_files_in_folder(UPLOAD_FOLDER, PROCESSED_FOLDER)

    # Compresser les fichiers traités pour le téléchargement
    zip_filename = "processed_files.zip"
    zip_filepath = os.path.join(PROCESSED_FOLDER, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for root, dirs, files in os.walk(PROCESSED_FOLDER):
            for file in files:
                if file != zip_filename:
                    zipf.write(os.path.join(root, file), arcname=file)

    return send_from_directory(PROCESSED_FOLDER, zip_filename, as_attachment=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return "Le fichier est trop volumineux ! La taille maximale autorisée est de {} Mo.".format(app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)), 413

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
