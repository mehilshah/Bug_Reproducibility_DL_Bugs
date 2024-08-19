from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def find_relevant_statements(report, bug_type):
    statements = report.split('. ')

    # No keywords for the 'Hyperparameter' category, as it is not one of the top 3 component information for any type of bugs.
    category_keywords = {
       'Data': ["data shape", "data type", "data distribution"],
       'Model': ["neural network architecture", "layers", "neurons", "activation functions"],
       'Logs': ["error logs", "diagnostic messages"],
       'Code': ["training code", "evaluation script"]
    }

    bug_categories = {
        'Training': ['Code', 'Logs', 'Data'],
        'Model': ['Logs', 'Model', 'Code'],
        'Tensor': ['Data', 'Logs', 'Code'],
        'API': ['Logs', 'Code', 'Model']
    }
    relevant_categories = bug_categories.get(bug_type, [])
    filtered_category_keywords = {k: category_keywords[k] for k in relevant_categories}

    category_relevance = {key: [] for key in filtered_category_keywords.keys()}

    for statement in statements:
        statement_embedding = embed_text(statement)

        for category, keywords in filtered_category_keywords.items():
            max_similarity = 0
            relevant_sentence = None

            for keyword in keywords:
                keyword_embedding = embed_text(keyword)
                similarity = 1 - cosine(statement_embedding.flatten(), keyword_embedding.flatten())

                if similarity > max_similarity:
                    max_similarity = similarity
                    relevant_sentence = statement

            if relevant_sentence:
                category_relevance[category].append((relevant_sentence, max_similarity))

    for category in category_relevance:
        category_relevance[category].sort(key=lambda x: x[1], reverse=True)
        category_relevance[category] = category_relevance[category][:3]

    return category_relevance

def read_files_and_analyze(relevant_path, included_extensions):
    file_names = [fn for fn in os.listdir(relevant_path)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    results = {}
    for file_name in file_names:
        bug_type = extract_bug_type(file_name)
        file_path = os.path.join(relevant_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            bug_report_text = file.read()
        analysis_results = find_relevant_statements(bug_report_text, bug_type)
        results[file_name] = analysis_results
    return results

relevant_path = "."
included_extensions = ['txt']
analysis_results = read_files_and_analyze(relevant_path, included_extensions)

with open('hint_formulation.txt', 'w') as file:
     file.write(json.dumps(analysis_results))