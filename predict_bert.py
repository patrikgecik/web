import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.nn.functional import softmax

# Definícia modelu
class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(self.dropout(pooled))

# Načítanie tokenizeru
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Načítanie modelov (len raz)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_en = BERTClassifier()
model_sk = BERTClassifier()

model_en.load_state_dict(torch.load("bert_model.pth", map_location=device))
model_sk.load_state_dict(torch.load("best_bert_model_sk.pth", map_location=device))

model_en.to(device).eval()
model_sk.to(device).eval()

# Funkcia na predikciu
def predict(text, lang="sk"):
    labels = ["Netoxická", "Toxická"] if lang == "sk" else ["Non-toxic", "Toxic"]
    model = model_sk if lang == "sk" else model_en

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device)
    }

    with torch.no_grad():
        output = model(**inputs)
        probs = softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    return f"{labels[pred]} ({probs[pred].item():.2%})"
