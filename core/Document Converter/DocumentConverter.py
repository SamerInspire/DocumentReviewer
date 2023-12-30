import os
import spacy
from sklearn import metrics
from sklearn.svm import SVC
import joblib
import fitz  # PyMuPDF
from farasa.pos import FarasaPOSTagger
from farasa.ner import FarasaNamedEntityRecognizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from docx import Document
import docx2txt
# Load the English NLP model from spaCy
nlp_en = spacy.load("en_core_web_sm")


# Function to extract text from a CV (resume)
def extract_text_from_cv(file_path, language="en"):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == ".docx":
        return extract_text_from_docx(file_path, language)
    elif file_extension == ".doc":
        return extract_text_from_doc(file_path, language)
    elif file_extension == ".pdf":
        return extract_text_from_pdf(file_path, language)
    elif file_extension == ".txt":
        return extract_text_from_txt(file_path, language)
    else:
        raise ValueError("Unsupported file type")


# Function to extract text from a .docx file
def extract_text_from_docx(file_path, language="en"):
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    if language == "ar":
        text = translate_to_arabic(text)
    return text


# Function to extract text from a .doc file
def extract_text_from_doc(file_path, language="en"):
    text = docx2txt.process(file_path)
    if language == "ar":
        text = translate_to_arabic(text)
    return text


# Function to extract text from a PDF file
def extract_text_from_pdf(file_path, language="en"):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    if language == "ar":
        text = translate_to_arabic(text)
    return text


# Function to extract text from a .txt file
def extract_text_from_txt(file_path, language="en"):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        if language == "ar":
            text = translate_to_arabic(text)
    return text


# Translate English text to Arabic using Farasa
def translate_to_arabic(text):
    pos_tagger = FarasaPOSTagger(interactive=True)
    ner_tagger = FarasaNamedEntityRecognizer(interactive=True)
    arabic_text = []
    for sentence in text.split('\n'):
        if sentence.strip():
            pos_tags = pos_tagger.tag(sentence)
            ner_tags = ner_tagger.tag(sentence)
            arabic_text.append(" ".join(
                [f"{token}/{pos}/{ner}" for token, pos, ner in zip(pos_tags['term'], pos_tags['pos'], ner_tags['ne'])]))
    return "\n".join(arabic_text)


# Function to check if CV matches the major and contains all sections
def check_cv_match(model, cv_text):
    # Predict the label using the model
    predicted_label = model.predict([cv_text])[0]

    # Extract potential standards from the new CV
    identified_standards = extract_standards_from_text(cv_text)

    # Check if the provided CV matches the major
    if predicted_label == 1 and identified_standards:
        return True
    else:
        return False


# Function to extract potential standards (you may need to customize this)
def extract_standards_from_text(text):
    doc = nlp_en(text)
    # Extract entities that might represent standards (customize based on your data)
    standards = [ent.text for ent in doc.ents if ent.label_ == "STANDARD"]
    return standards


# Mock dataset for training the model
training_data = []
with os.scandir('../../Resources/Positive/') as entries:
    for entry in entries:
        print(entry)
        training_data.append({"file_path": '../../Resources/Positive/' + str(entry.name), "label": 1})

with os.scandir('../../Resources/Negative/') as entries:
    for entry in entries:
        print(entry.name)
        training_data.append({"file_path": '../../Resources/Negative/' + str(entry.name), "label": 0})

print(training_data)

# Prepare data for training
texts = [extract_text_from_cv(item["file_path"]) for item in training_data]
labels = [item["label"] for item in training_data]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define a pipeline with TfidfVectorizer and Support Vector Machine classifier
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), SVC(kernel='linear', C=1))

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file (you can use joblib or pickle for this)
model_filename = "../AI Models/cv_standards_model.joblib"
joblib.dump(model, model_filename)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define a pipeline with TfidfVectorizer and RandomForestClassifier
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42))

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# ... (save the model and use it for testing)

# Test the model on a new CV
new_cv_path = "../../Resources/match cv 1.txt"
new_text = extract_text_from_cv(new_cv_path, language="en")
print('new_text ---> ',new_text)
predicted_label = model.predict([new_text])[0]

# Extract potential standards from the new CV
identified_standards = extract_standards_from_text(new_text)

print(f"Predicted Label: {predicted_label}")
print(f"Identified Standards: {identified_standards}")

# Check if the provided document matches the model with accuracy
if predicted_label == 1:
    print("The document is compliant with the model.")
else:
    print("The document is non-compliant with the model.")


