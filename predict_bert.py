import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.nn.functional import softmax
from huggingface_hub import hf_hub_download

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
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
except Exception as e:
    print("❌ Chyba pri načítaní tokenizeru:", e)
    exit(1)

# Sťahovanie modelu
try:
    model_path = hf_hub_download(
        repo_id="patrikgecik/best_bert_model_sk",
        filename="best_bert_model_sk.pth"
    )
except Exception as e:
    print("❌ Chyba pri sťahovaní modelu:", e)
    exit(1)

# Inicializácia modelu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier()
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except Exception as e:
    print("❌ Chyba pri načítaní modelu:", e)
    exit(1)

model.to(device)
model.eval()

def predict(text, lang="sk"):
    labels = ["Netoxická", "Toxická"]

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
