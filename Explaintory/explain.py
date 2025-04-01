from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT",token=token)
model = AutoModel.from_pretrained("medicalai/ClinicalBERT",token=token)
