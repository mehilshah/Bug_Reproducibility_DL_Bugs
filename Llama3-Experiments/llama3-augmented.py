import transformers
import torch
import csv
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
model_id = 'meta-llama/Meta-Llama-3-70B'
pipeline = transformers.pipeline('text-generation', model=model_id)
logging.info(f"Pipeline for model {model_id} created successfully.")

info_and_actions = {
    "Training Bug": {
        "info": ["Code Snippets", "Data", "Logs"],
        "actions": ["Input Data Generation", "Import Addition", "Compiler Error Resolution", "Dataset Procurement", "Hyperparameter Initialization"]
    },
    "Model Bug": {
        "info": ["Logs", "Code Snippets", "Model Details"],
        "actions": ["Hyperparameter Initialization", "Dataset Procurement", "Compiler Error Resolution", "Import Addition", "Neural Network Construction"]
    },
    "Tensor and Input Bug": {
        "info": ["Data", "Logs", "Code Snippets"],
        "actions": ["Hyperparameter Initialization", "Input Data Generation", "Import Addition", "Dataset Procurement", "Obsolete Parameter Removal"]
    },
    "API Bug": {
        "info": ["Logs", "Code Snippets", "Model Details"],
        "actions": ["Input Data Generation", "Hyperparameter Initialization", "Import Addition", "Logging", "Obsolete Parameter Removal"]
    }
}

def process_bugs(file_path):
    responses = []
    logging.info(f"Opening file {file_path} for processing.")
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            bug_type = row['Type of Bug']
            prompt = construct_prompt(row['Code'], row['Text'], bug_type)
            logging.debug(f"Generated prompt for Bug ID {row['Bug ID']}.")
            response = pipeline(prompt)
            logging.debug(f"Received response for Bug ID {row['Bug ID']}.")
            responses.append({'Bug ID': row['Bug ID'], 'Response': response[0]['generated_text']})
    logging.info(f"Processed all bugs from file {file_path}.")
    return responses

def construct_prompt(code, description, bug_type):
    info = info_and_actions[bug_type]["info"]
    actions = info_and_actions[bug_type]["actions"]
    prompt = f"### Bug Report: {description}\n### Code Snippet:\n{code}\n### Focus on Information:\n- " + "\n- ".join(info) + "\n### Recommended Edit Actions:\n- " + "\n- ".join(actions) + "\nGenerate a Python code snippet that helps me reproduce the bug."
    return prompt

def write_responses(file_path, data):
    logging.info(f"Writing responses to file {file_path}.")
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Bug ID', 'Response']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item)
            logging.debug(f"Written response for Bug ID {item['Bug ID']}.")
    logging.info(f"All responses have been written to {file_path}.")

input_csv = 'Cleaned_LLaMA_Bugs.csv'
output_csv = 'Augmented_LLaMA_Responses.csv'

logging.info("Starting bug processing.")
bug_responses = process_bugs(input_csv)
logging.info("Bug processing completed.")

logging.info("Starting to write responses.")
write_responses(output_csv, bug_responses)
logging.info("Responses writing completed.")

print(f"Responses to bugs saved to '{output_csv}'")
