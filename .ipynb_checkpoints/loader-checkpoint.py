import os

def load_data(data_path):
    documents = []
    labels = []

    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)

        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(category_path, filename)

                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                documents.append(text)
                labels.append(category)

    return documents, labels