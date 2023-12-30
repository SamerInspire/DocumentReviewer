import spacy
from spacy.training.example import Example
import random
import docx2txt


def extract_text_from_cv(file_path, language="en"):
    if file_path.lower().endswith('.txt'):
        # Handle plain text files
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    elif file_path.lower().endswith(('.doc', '.docx')):
        # Handle Word documents
        text = docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file format. Only .txt, .doc, and .docx are supported.")

    return text

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
def extract_it_keywords(model):
    # Extract IT-related keywords from entities labeled as "IT_SECTION" in the model's training data
    it_keywords = set()
    for text, annotations in training_data:
        if "IT_SECTION" in annotations['entities']:
            doc = model(text)
            it_keywords.update([ent.text for ent in doc.ents if ent.label_ == "IT_SECTION"])
    return it_keywords

def check_related_to_field(nlp, cv_path, it_keywords):
    # Test the model on a new CV
    new_text = extract_text_from_cv(cv_path, language="en")
    doc = nlp(new_text)
    print('new_text ==-=> ',cv_path,'\n ',new_text)
    # Extract entities from the document
    entities = [ent.label_ for ent in doc.ents]

    # Check if specific entities are related to the trained data field
    # it_section_related = "IT_SECTION" in entities
    # standard_related = "STANDARD" in entities
    education_related = "EDUCATION" in entities
    experience_related = "EXPERIENCE" in entities
    # Extract education and experience entities
    education_entities = [ent.text for ent in doc.ents if ent.label_ == "EDUCATION"]
    experience_entities = [ent.text for ent in doc.ents if ent.label_ == "EXPERIENCE"]

    # Check if there are relevant entities in both education and experience sections
    it_education = any(keyword in edu for edu in education_entities for keyword in it_keywords)
    it_experience = any(keyword in exp for exp in experience_entities for keyword in it_keywords)

    # Check if there are no relevant entities in either education or experience sections
    if not it_education and not it_experience:
        return "False"
    # Check if there are relevant entities in both education and experience sections
    elif it_education and it_experience:
        return "True"
    # Otherwise, there is some ambiguity or a partial match
    else:
        return "Maybe"


# Updated training data
training_data = [
    ("Master's in Computer Science", {"entities": [(0, 32, "EDUCATION")]}),
    ("Bachelor's in Information Technology", {"entities": [(0, 41, "EDUCATION")]}),
    ("High School Diploma", {"entities": [(0, 22, "EDUCATION")]}),
    ("Senior Software Engineer", {"entities": [(0, 29, "EXPERIENCE")]}),
    ("Software Developer", {"entities": [(0, 31, "EXPERIENCE")]}),
    ("System Administrator", {"entities": [(0, 39, "EXPERIENCE")]}),
    ("Network Engineer", {"entities": [(0, 37, "EXPERIENCE")]}),
    ("This CV adheres to ISO 9001 standards and follows industry best practices.",
     {"entities": [(49, 70, "STANDARD"), (21, 32, "IT_SECTION")]}),
    ("Ph.D. in Computer Science", {"entities": [(0, 36, "EDUCATION")]}),
    ("Junior Developer at UVW Tech", {"entities": [(0, 32, "EXPERIENCE")]}),
    ("Associate's in Software Engineering", {"entities": [(0, 45, "EDUCATION")]}),
    ("Lead Software Engineer", {"entities": [(0, 37, "EXPERIENCE")]}),
]
# Development data for evaluation
dev_data = [
    ("Ph.D. in Computer Science", {"entities": [(0, 36, "EDUCATION")]}),
    ("Junior Developer", {"entities": [(0, 32, "EXPERIENCE")]}),
    ("Associate's in Software Engineering", {"entities": [(0, 45, "EDUCATION")]}),
    ("Lead Software Engineer", {"entities": [(0, 37, "EXPERIENCE")]}),
    ("Machine Learning Engineer", {"entities": [(0, 25, "EXPERIENCE")]}),  # Added IT-related experience
]

nlp = spacy.load("../AI Models/it_ner_model")

# Train the NER model
train_ner_model(nlp, training_data, dev_data)



# Load the trained model
nlp = spacy.load("../AI Models/it_ner_model")

# Test with three different CVs
cv_path_true = "../../Resources/match cv 1.txt"
cv_path_maybe = "../../Resources/maybe match cv 1.txt"
cv_path_false = "../../Resources/not match cv 1.txt"
it_keywords = extract_it_keywords(spacy.blank("en"))

result_true = check_related_to_field(nlp, cv_path_true, it_keywords)
result_maybe = check_related_to_field(nlp, cv_path_maybe, it_keywords)
result_false = check_related_to_field(nlp, cv_path_false, it_keywords)
sample_cv = "I have a Master's in Computer Science from XYZ University in 2020. Previously, I worked as a Senior Software Engineer at ABC Tech from 2021 to the present."
doc = nlp(sample_cv)
# Load the trained model
nlp = spacy.load("../AI Models/it_ner_model")

print("Result for True CV:", result_true)
print("Result for Maybe CV:", result_maybe)
print("Result for False CV:", result_false)
