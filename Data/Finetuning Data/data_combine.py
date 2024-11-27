import json
import pandas as pd 

linear_algebra_output = 'linear_algebra_finetuning_chat.jsonl'
abstract_algebra_output = 'abstract_algebra_finetuning_chat.jsonl'
combined_output = 'combined_finetuning_data.jsonl'

# Combine the datasets into one file for convenience
with open(combined_output, 'w') as combined_file:
    for file_name in [linear_algebra_output, abstract_algebra_output]:
        with open(file_name, 'r') as data_file:
            for line in data_file:
                combined_file.write(line)


csv_files = [
    'Linear Algebra Calculation Data 1000.csv',
    'Sythetic Abstract Algebra Data 3000.csv',
    'Sythetic Linear Algebra Data 5000.csv'
]

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df['text'] = "Question: " + df['Question'] + " Answer: " + df['Answer']
    dfs.append(df[['text']])  # Only keep the "text" column

combined_df = pd.concat(dfs)
combined_df.dropna(inplace=True)
combined_df.to_csv("combined_finetuning_data.csv", index=False)
