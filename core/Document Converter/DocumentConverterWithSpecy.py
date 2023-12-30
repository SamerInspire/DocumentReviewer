import spacy
from spacy.training.example import Example
import random
import docx2txt
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_cv(file_path, language="en"):
    if file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    elif file_path.lower().endswith(('.doc', '.docx')):
        text = extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Only .txt, .doc, .docx, and .pdf are supported.")
    return text

def load_training_data_from_folder(folder_path, label):
    training_data = []
    folder = Path(folder_path)
    for file_path in folder.glob("*.*"):
        text = extract_text_from_cv(file_path)
        training_data.append((text, {"entities": [(0, len(text), label)]}))
    return training_data


def train_ner_model(nlp, train_data, dev_data, output_dir="it_ner_model", n_iter=20):
    for epoch in range(n_iter):
        random.shuffle(train_data)
        losses = 0.0
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            gold_dict = {"entities": []}
            for start, end, label in annotations['entities']:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    gold_dict['entities'].append((span.start_char, span.end_char, label))
            example = Example.from_dict(doc, gold_dict)
            losses += nlp.update([example], drop=0.5).get("ner", 0.0)

        # Evaluate on the development set
        dev_loss = 0.0
        for text, annotations in dev_data:
            doc = nlp.make_doc(text)
            gold_dict = {"entities": []}
            for start, end, label in annotations['entities']:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    gold_dict['entities'].append((span.start_char, span.end_char, label))
            example = Example.from_dict(doc, gold_dict)
            dev_loss += nlp.evaluate([example]).get("ner", 0.0)

        print(f"Epoch {epoch + 1}/{n_iter}, Training Loss: {losses:.2f}, Dev Loss: {dev_loss:.2f}")

    # Save the trained model to a directory
    nlp.to_disk(output_dir)
# Load negative training data
negative_training_data = load_training_data_from_folder("./../Resources/Negative", "NEGATIVE")

# Load positive training data
positive_training_data = load_training_data_from_folder("./../Resources/Positive", "POSITIVE")

# Corrected training data
training_data = [
    ("Master's in Computer Science", {"entities": [(0, 27, "EDUCATION")]}),
    ("Bachelor's in Information Technology", {"entities": [(0, 41, "EDUCATION")]}),
    ("High School Diploma", {"entities": [(0, 19, "EDUCATION")]}),
    ("Senior Software Engineer", {"entities": [(0, 23, "EXPERIENCE")]}),
    ("Software Developer", {"entities": [(0, 17, "EXPERIENCE")]}),
    ("System Administrator", {"entities": [(0, 21, "EXPERIENCE")]}),
    ("Network Engineer", {"entities": [(0, 19, "EXPERIENCE")]}),
    ("This CV adheres to ISO 9001 standards and follows industry best practices.",
     {"entities": [(49, 70, "STANDARD"), (21, 32, "IT_SECTION")]}),
    ("Ph.D. in Computer Science", {"entities": [(0, 26, "EDUCATION")]}),
    ("Junior Developer at UVW Tech", {"entities": [(0, 25, "EXPERIENCE")]}),
    ("Associate's in Software Engineering", {"entities": [(0, 34, "EDUCATION")]}),
    ("Lead Software Engineer", {"entities": [(0, 21, "EXPERIENCE")]}),
]

# Development data for evaluation
dev_data = [
    ("Computer Science", {"entities": [(0, 36, "EDUCATION")]}),
    ("Junior Developer", {"entities": [(0, 32, "EXPERIENCE")]}),
    ("Software Engineering", {"entities": [(0, 45, "EDUCATION")]}),
    ("Lead Software Engineer", {"entities": [(0, 37, "EXPERIENCE")]}),
    ("Machine Learning Engineer", {"entities": [(0, 25, "EXPERIENCE")]}),  # Added IT-related experience
]
# training_data = positive_training_data + negative_training_data

nlp = spacy.load("../AI Models/it_ner_model")

# Train the NER model
train_ner_model(nlp, training_data, dev_data)
# Shuffle the training data
random.shuffle(training_data)

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Add the NER component to the pipeline
ner = nlp.get_pipe("ner")

# Add your labels to the NER component
for _, annotations in training_data:
    for ent in annotations.get("entities", []):
        ner.add_label(ent[2])

# Training loop
for _ in range(10):  # You can adjust the number of iterations
    random.shuffle(training_data)
    for text, annotations in training_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example])

# Save the trained model
nlp.to_disk("it_ner_model")



def extract_all_entities(model):
    # Extract all entity labels from the NER component
    all_entities = set()
    ner = model.get_pipe("ner")
    for label in ner.labels:
        all_entities.add(label)
    return all_entities
def check_related_to_field(model, cv_path, it_keywords):
    text = extract_text_from_cv(cv_path)
    doc = model(text)
    print('Entities in document:', [(ent.text, ent.label_) for ent in doc.ents])

    entities = [ent.label_ for ent in doc.ents]
    education_related = "EDUCATION" in entities
    experience_related = "EXPERIENCE" in entities
    education_entities = [ent.text for ent in doc.ents if ent.label_ == "EDUCATION"]
    experience_entities = [ent.text for ent in doc.ents if ent.label_ == "EXPERIENCE"]
    it_education = any(keyword in edu for edu in education_entities for keyword in it_keywords)
    it_experience = any(keyword in exp for exp in experience_entities for keyword in it_keywords)

    if not it_education and not it_experience:
        return "False"
    elif it_education and it_experience:
        return "True"
    else:
        return "Maybe"

# Test with the three files
cv_path_true = "../../Resources/match cv 1.txt"
cv_path_maybe = "../../Resources/maybe match cv 1.txt"
cv_path_false = "../../Resources/not match cv 1.txt"


# Extract IT-related keywords from entities labeled as "IT_SECTION" in the model's training data
it_keywords = extract_all_entities(nlp)
print('it_keywords ===> ',it_keywords)
# Check the results for each CV
result_true = check_related_to_field(nlp, cv_path_true, it_keywords)
result_false = check_related_to_field(nlp, cv_path_false, it_keywords)
result_maybe = check_related_to_field(nlp, cv_path_maybe, it_keywords)

print(f"Result for True CV ({cv_path_true}):", result_true)
print(f"Result for False CV ({cv_path_false}):", result_false)
print(f"Result for Maybe CV ({cv_path_maybe}):", result_maybe)

doc = nlp("Senior Software Engineer")
print([(ent.text, ent.label_) for ent in doc.ents])
