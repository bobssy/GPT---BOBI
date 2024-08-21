import os
import logging
import threading
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import Dataset
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from docx import Document
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import torch
from functools import lru_cache
import pytesseract
from PIL import Image
import io
import optuna

# Настройка на логване
logging.basicConfig(level=logging.INFO)

# Проверка за наличието на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Зареждане на GPT-Neo модел
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)

# Добавяне на padding token ако не съществува
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Път до обработените файлове
processed_files_path = 'processed_files.txt'

# Генериране на текст с параметри за креативност
@lru_cache(maxsize=100)
def generate_text(prompt, max_length=2048, temperature=0.6, top_k=40):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature,
            top_k=top_k,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Generated text: {response}")
        return response
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        return "Sorry, I couldn't generate a response."

# Функции за извличане на текст
# OCR функция за извличане на текст от изображения в PDF файлове
def extract_text_with_ocr(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logging.error(f"Error extracting text with OCR from {pdf_path}: {e}")
    return None

# Извличане на текст от PDF файлове с PyMuPDF
def extract_text_from_pdf_with_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path} with PyMuPDF: {e}")
    return None

# Извличане на текст от PDF файлове с pdfminer
def extract_text_from_pdf_with_pdfminer(pdf_path):
    try:
        return pdfminer_extract_text(pdf_path)
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path} with pdfminer: {e}")
    return None

# Извличане на текст от PDF файлове
def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text_from_pdf_with_pymupdf(pdf_path)
        if not text:
            logging.warning(f"Failed to extract text from {pdf_path} with PyMuPDF. Trying pdfminer...")
            text = extract_text_from_pdf_with_pdfminer(pdf_path)
        if not text:
            logging.warning(f"Failed to extract text from {pdf_path} with pdfminer. Trying OCR...")
            text = extract_text_with_ocr(pdf_path)
        if not text:
            logging.error(f"Failed to extract text from {pdf_path} using all methods.")
        return text
    except Exception as e:
        logging.error(f"Unexpected error extracting text from {pdf_path}: {e}")
    return None

# Извличане на текст от Word файлове
def extract_text_from_word(doc_path):
    try:
        doc = Document(doc_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {doc_path} (Word): {e}")
    return None

# Извличане на текст от други формати (HTML, TXT)
def extract_text_from_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text()
    except Exception as e:
        logging.error(f"Error extracting text from {file_path} (HTML): {e}")
    return None

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error extracting text from {file_path} (TXT): {e}")
    return None

# Получаване на всички файлове в директория
def get_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.pdf', '.html', '.txt', '.docx'))]

# Зареждане на датасет
def load_dataset(file_paths):
    texts = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                futures.append(executor.submit(extract_text_from_pdf, file_path))
            elif file_path.endswith('.docx'):
                futures.append(executor.submit(extract_text_from_word, file_path))
            elif file_path.endswith('.html'):
                futures.append(executor.submit(extract_text_from_html, file_path))
            elif file_path.endswith('.txt'):
                futures.append(executor.submit(extract_text_from_txt, file_path))
            else:
                logging.warning(f"Unsupported file format: {file_path}")

        for future in futures:
            text = future.result()
            if text:
                texts.append(text)

    # Валидиране на текстовете
    valid_texts = [text for text in texts if len(text.strip()) > 0]

    dataset = Dataset.from_dict({"text": valid_texts})
    return dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128), batched=True)

# Създаване на Data Collator
def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

# Проверка на новите файлове
def get_processed_files():
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r', encoding='utf-8', errors='ignore') as file:
            return set(line.strip() for line in file)
    return set()

def save_processed_files(files):
    with open(processed_files_path, 'a', encoding='utf-8') as file:
        for file_path in files:
            file.write(file_path + '\n')

def get_new_files():
    processed_files = get_processed_files()
    all_files = get_files('D:\\learn')
    new_files = [file for file in all_files if file not in processed_files]
    return new_files


# Оптимизация на хиперпараметрите с Optuna
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16])
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results['eval_loss']

# Самообучение на модела
def train_model():
    global training_progress, training_in_progress
    new_files = get_new_files()

    if not new_files:
        logging.info("No new files to process.")
        training_in_progress = False
        return

    try:
        train_dataset = load_dataset(new_files)
        data_collator = create_data_collator(tokenizer)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)

        logging.info("Saving model...")
        model.save_pretrained('trained_model')
        tokenizer.save_pretrained('trained_model')

        logging.info("Training completed successfully.")
        save_processed_files(new_files)

    except Exception as e:
        logging.error(f"Error during training: {e}")

    finally:
        training_in_progress = False

# Функция за показване на "BOBI GPT 1.2" като скрийнсейвър
def display_console_message():
    message = """
____   ___  ____ ___    ____ ____ _____   _   ____  
| __ ) / _ \| __ )_ _|  / ___|  _ \_   _| / | |___ \ 
|  _ \| | | |  _ \| |  | |  _| |_) || |   | |   __) |
| |_) | |_| | |_) | |  | |_| |  __/ | |   | |_ / __/ 
|____/ \___/|____/___|  \____|_|    |_|   |_(_)_____|
    """
    while True:
        if not training_in_progress:
            print(message)
        time.sleep(300)

# Добавяне на нова глобална променлива за контрол на скрийнсейвъра
training_in_progress = False

# Стартиране на скрийнсейвъра в отделен поток
threading.Thread(target=display_console_message, daemon=True).start()

# Инициализация на Flask приложението
app = Flask(__name__)

# Корс конфигурация
CORS(app)

# Статус на обучението
training_in_progress = False
training_progress = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            logging.error('No JSON data received')
            return jsonify({'response': 'No data received.'}), 400

        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'response': 'No prompt provided.'}), 400

        response = generate_text(prompt)
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        return jsonify({'response': 'Error during operation.'}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        upload_folder = 'D:\\learn'
        os.makedirs(upload_folder, exist_ok=True)
        
        for file in files:
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

        return jsonify({'message': 'Files uploaded successfully'})
    except Exception as e:
        logging.error(f"Error in /upload endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_in_progress

    if training_in_progress:
        return jsonify({'status': 'Training already in progress.'}), 400

    training_in_progress = True
    train_model()
    return jsonify({'status': 'Training started.'})

if __name__ == '__main__':
    app.run(debug=True)
