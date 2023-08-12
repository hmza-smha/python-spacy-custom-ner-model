# How to train a custom model with a custom NER in Python SpaCy library?

> Pre-requests: 
Prepare your training data using [ner-annotator](https://tecoholic.github.io/ner-annotator/).

#### Install SpaCy
```
pip install -U spacy -q 
```
##### Check installation successfully
```
python -m spacy info
```

#### Run this code 
```python
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json

# load a new spacy model
# use a pre definded model using { python -m spacy download en_core_web_sm } 
# then => nlp = spacy.load("en_core_web_sm")
# but we are creating a custom one
# use 'ar' & encoding="utf8" for arabic language
nlp = spacy.blank("en") 

db = DocBin() # create a DocBin object

f = open('training_data.json') # the path for your training data

TRAIN_DATA = json.load(f)

for text, annot in tqdm(TRAIN_DATA['annotations']): 
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents 
    db.add(doc)

db.to_disk("./training_data.spacy") # save the docbin object
```

#### Load config file 
```
// use 'ar' for arabic language
python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency
```
> A new file with .spacy extention file created

#### Train your data
```
python -m spacy train config.cfg --output ./ --paths.train ./training_data.spacy --paths.dev ./training_data.spacy
```

> Note: You have a trained model now, so no need for the previous code, you just need the following

#### Load your trained model 
```python
nlp_ner = spacy.load("/content/model-best") # path for the model
```
### Test you model
```python
doc = nlp_ner('''Your Text''')
for ent in doc.ents:
    print(ent.label_)
```