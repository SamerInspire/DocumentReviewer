import spacy
from pathlib import Path

from DocumentConverterWithSpecy import extract_text_from_cv


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

# Load the trained model
nlp = spacy.load("../AI Models/it_ner_model")

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