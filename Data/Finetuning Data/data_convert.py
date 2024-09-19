import pandas as pd
import json

def prepare_and_save_data(file_paths, output_file, combine=False,chat_format=False):
    # Since we have two datasets for linear algera, we have to combine them into one for convenience
    if combine:
        data_frames = [pd.read_csv(file) for file in file_paths]
        data = pd.concat(data_frames, ignore_index=True)
    else:
        data = pd.read_csv(file_paths[0])
    
    # Rename columns to match the GPT model finetuning requirements
    data = data.rename(columns={'Question': 'prompt', 'Answer': 'completion'})
    data['completion'] = data['completion'].apply(lambda x: str(x) + '\n')
    
    # Convert the DataFrame to a list of dictionaries
    json_data = data.to_dict(orient='records')
    
    # Write to a JSONL file
    with open(output_file, 'w') as f:
        for entry in json_data:
            if chat_format:
                # Reformat data to the required chat format to meet GPT-3.5-turbo and GPT-4o finetuning 
                chat_entry = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant specializing in abstract algebra."},
                        {"role": "user", "content": entry['prompt']},
                        {"role": "assistant", "content": entry['completion']}
                    ]
                }
                json.dump(chat_entry, f)
            else:
                # Non-chat format (prompt-completion format)
                json.dump(entry, f)
            f.write('\n')


# Access the file paths
linear_algebra_files = ["Linear Algebra Calculation Data 1000.csv", "Sythetic Linear Algebra Data 5000.csv"]
abstract_algebra_file = ["Sythetic Abstract Algebra Data 3000.csv"]

# Save the data to JSONL files
prepare_and_save_data(linear_algebra_files, 'linear_algebra_finetuning_chat.jsonl', combine=True, chat_format=True)
prepare_and_save_data(abstract_algebra_file, 'abstract_algebra_finetuning_chat.jsonl', chat_format=True)

