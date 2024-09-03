import pandas as pd
import datasets
import json

# Since the format of linear algebra qa data is different from our expectation, we need to eliminate the unwanted parts and save it in a csv file
def separate_qa(data):
    """
    Separates a list of strings containing question-answer pairs into two columns, removing any "<s> Level-1:" prefix from questions and "</s>" from answers

    Args:
        data: A list of strings, each containing a question-answer pair in the format "[INST] Question [/INST] Answer", potentially with a "<s> Level-1:" prefix before the question and "</s>" at the end of the answer

    Returns:
        A Pandas DataFrame with two columns: "Question" and "Answer".
    """

    questions = []
    answers = []

    for item in data:
        parts = item.split("[/INST]")
        question = parts[0].replace("[INST]", "").strip()  # Remove "[INST]" and extra spaces

        # 'Level' prefix is not necessary for the fine-tuning process
        question = question.replace("<s> Level-1:", "").strip()
        answer = parts[1].strip()  

        # Remove "</s>" from the answer
        answer = answer.replace("</s>", "").strip()

        questions.append(question)
        answers.append(answer)

    # Store the questions and answers in a DataFrame for uniformity
    df = pd.DataFrame({"Question": questions, "Answer": answers})
    return df


# Loading the linear algebra QA data from huggingface datasets
df = pd.read_parquet("hf://datasets/Likhi2003/linearalgebra_QA/data/train-00000-of-00001.parquet")
linear_algebra_qa = separate_qa(df["text"].tolist())
 
# Save the DataFrame to a CSV file
linear_algebra_qa.to_csv("linear_algebra_qa_test.csv", index=False)


# The TheoremQA dataset contains "Algebra" subfield, which requires human evaluation to extract linear algebra questions
# Extracting linear algebra part from the TheoremQA dataset
def filter_data_by_subfield(data, subfield):
  """
  Filters data based on the 'subfield' key in a list of dictionaries.

  Args:
      data: A list of dictionaries representing the data.
      subfield: The value of the 'subfield' key to filter by.

  Returns:
      A list of dictionaries where the 'subfield' key matches the given value.
  """
  filtered_data = [item for item in data if item.get('subfield') == subfield]
  return filtered_data

# https://github.com/wenhuchen/TheoremQA.git
data = json.load(open('theoremqa_test.json', 'r'))
filtered_data = filter_data_by_subfield(data, 'Algebra')

# Save the filtered data to a new JSON file
with open('linear_algebra_theorem_qa_test.json', 'w') as f:
    json.dump(filtered_data, f, indent=4) 



# Load MMLU abstract_algebra dataset, MMLU is a widely used benchmark for evaluating the performance of model's mathematical abilities.
dataset = datasets.load_dataset("lukaemon/mmlu", "abstract_algebra")

def transform_data(data):
  """
  Transforms the given data dictionary into a question-answer pair.

  Args:
      data: A dictionary containing 'input', 'A', 'B', 'C', 'D', and 'target' keys.

  Returns:
      A tuple containing the transformed question and answer strings.
  """

  question = data['input']
  answer = None  

  for key in ['A', 'B', 'C', 'D']:  # MMLU target key only contains "A", "B", "C", or "D", not exact answer
    if data['target'] == key:       # So, we need to extract the answer from the corresponding key for further evaluation
      answer = data[key]
  return question, answer

questions = []
answers = []

for i in range(len(dataset['test'])): 
    data = dataset['test'][i]
    question, answer = transform_data(data) # Transform the data into a question-answer pair
    questions.append(question)
    answers.append(answer)

# Transform extracted questions and answers into a DataFrame, and save it to a CSV file
abstract_algebra_qa_test = pd.DataFrame({'Question': questions, 'Answer': answers})
abstract_algebra_qa_test.to_csv("abstract_algebra_qa_test.csv", index=False)
