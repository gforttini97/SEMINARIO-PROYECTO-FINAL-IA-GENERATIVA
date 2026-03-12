# Documentación

from datasets import load_dataset

ds = load_dataset("microsoft/ms_marco", "v1.1")

documents = []

metadata = []

sample = ds["train"].select(range(2000))

for row in sample:
   
    query_id = row["query_id"]
    
    query = row["query"]

    texts = row["passages"]["passage_text"]
    selected_flags = row["passages"].get("is_selected", [0] * len(texts))

    for j, text in enumerate(texts):
        if text and text.strip() and selected_flags[j] == 1:
            documents.append(text)
            metadata.append({
                "query_id": query_id,
                "query": query,
                "passage_idx": j,
                "is_selected": selected_flags[j]
            })

print("Cantidad de textos extraídos:", len(documents))

print(documents[0][:1000])

print(metadata[0])

https://huggingface.co/settings/tokens
