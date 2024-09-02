import pandas as pd
import json

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
        # Split the string into question and answer parts based on the delimiters
        parts = item.split("[/INST]")
        question = parts[0].replace("[INST]", "").strip()  # Remove "[INST]" and extra spaces

        # Remove "<s> Level-1:" prefix if present
        question = question.replace("<s> Level-1:", "").strip()

        answer = parts[1].strip()  # Remove extra spaces

        # Remove "</s>" from the answer
        answer = answer.replace("</s>", "").strip()

        questions.append(question)
        answers.append(answer)

    # Create a DataFrame from the extracted questions and answers
    df = pd.DataFrame({"Question": questions, "Answer": answers})
    return df


# Loading the data from huggingface datasets
df = pd.read_parquet("hf://datasets/Likhi2003/linearalgebra_QA/data/train-00000-of-00001.parquet")
linear_algebra_qa = separate_qa(df["text"].tolist())
 
# Save the DataFrame to a CSV file
linear_algebra_qa.to_csv("linear_algebra_qa_test.csv", index=False)


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

# Filter the data for the 'Algebra' subfield
# https://github.com/wenhuchen/TheoremQA.git
data = json.load(open('theoremqa_test.json', 'r'))
filtered_data = filter_data_by_subfield(data, 'Algebra')

# Save the filtered data to a new JSON file
with open('linear_algebra_theorem_qa_test.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)  # indent for better readability
