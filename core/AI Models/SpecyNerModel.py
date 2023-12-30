import spacy
import random

# Sample dataset with synthetic data
training_data = [
    ("This CV adheres to ISO 9001 standards and follows industry best practices.", {"entities": [(21, 32, "STANDARD"), (49, 70, "IT_SECTION")]}),
    ("Experienced software engineer with expertise in full-stack web development.", {"entities": [(22, 41, "IT_SECTION")]}),
    # Add more examples...
]

# Load the spaCy model
nlp = spacy.blank("en")

# Add the NER pipeline
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner, last=True)

# Add custom labels for IT_SECTION and STANDARD
ner.add_label("IT_SECTION")
ner.add_label("STANDARD")

# Begin training
nlp.begin_training()

# Train for 10 epochs (you may need more epochs for a real dataset)
for _ in range(10):
    random.shuffle(training_data)
    for text, annotations in training_data:
        example = spacy.training.Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example])

# Save the trained model to a file
nlp.to_disk("it_ner_model")