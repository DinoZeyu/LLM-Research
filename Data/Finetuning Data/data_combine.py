import json

linear_algebra_output = 'linear_algebra_finetuning_chat.jsonl'
abstract_algebra_output = 'abstract_algebra_finetuning_chat.jsonl'
combined_output = 'combined_finetuning_data.jsonl'

# Combine the datasets into one file for convenience
with open(combined_output, 'w') as combined_file:
    for file_name in [linear_algebra_output, abstract_algebra_output]:
        with open(file_name, 'r') as data_file:
            for line in data_file:
                combined_file.write(line)
