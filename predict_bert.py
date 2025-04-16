import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.nn.functional import softmax
import sys
import io
import os



sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Vstupné argumenty
if len(sys.argv) < 2:
    print("⚠️ Zadaj vetu ako argument.")
    sys.exit(1)

text = sys.argv[1]
lang = sys.argv[2] if len(sys.argv) > 2 else "sk"  # predvolený jazyk: SK

# 2. Nastavenie podľa jazyka
if lang == "en":
    model_path = "bert_model.pth"
    labels = ["Non-toxic", "Toxic"]
else:
    model_path = "best_bert_model_sk.pth"
    labels = ["Netoxická", "Toxická"]

# 3. Načítanie tokenizeru
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 4. Definícia modelu (rovnaký pre oba jazyky)
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

# 5. Načítanie modelu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier()

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except RuntimeError as e:
    print("❌ Chyba: Model sa nepodarilo načítať. Skontroluj typ a architektúru modelu.")
    print(str(e))
    sys.exit(1)

model.to(device)
model.eval()

# 6. Tokenizácia vstupu (iba potrebné parametre)
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

# 7. Predikcia
with torch.no_grad():
    output = model(**inputs)
    probs = softmax(output, dim=1)[0]
    pred = torch.argmax(probs).item()

# 8. Výstup
print(f"\n➡️ Výsledok: {labels[pred]} ({probs[pred].item():.2%})")
print(f"🟢 {labels[0]}: {probs[0].item():.2%}")
print(f"🔴 {labels[1]}: {probs[1].item():.2%}")
