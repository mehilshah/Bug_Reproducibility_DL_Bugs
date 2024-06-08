import transformers
import torch
import csv
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
model_id = 'meta-llama/Meta-Llama-3-70B'
pipeline = transformers.pipeline('text-generation', model=model_id)
logging.info(f"Pipeline for model {model_id} created successfully.")

def process_bugs(file_path):
    responses = []
    logging.info(f"Opening file {file_path} for processing.")
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompt = f"Code:\n{row['Code']}\nDescription:\n{row['Text']}\n## Generate a code snippet to reproduce the bug."
            logging.debug(f"Generated prompt for Bug ID {row['Bug ID']}.")
            response = pipeline(prompt)
            logging.debug(f"Received response for Bug ID {row['Bug ID']}.")
            responses.append({'Bug ID': row['Bug ID'], 'Response': response[0]['generated_text']})
    logging.info(f"Processed all bugs from file {file_path}.")
    return responses

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
output_csv = 'LLaMA_Responses.csv'

logging.info("Starting bug processing.")
bug_responses = process_bugs(input_csv)
logging.info("Bug processing completed.")

logging.info("Starting to write responses.")
write_responses(output_csv, bug_responses)
logging.info("Responses writing completed.")

print(f"Responses to bugs saved to '{output_csv}'")