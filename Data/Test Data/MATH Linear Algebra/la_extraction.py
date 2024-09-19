import json
import random
import os
import re

# Paths to the folders
determinant_folder_path = "determinant"
eigenvalue_folder_path = "eigenvalues"

# Get all .txt files in both folders
determinant_file_list = [f for f in os.listdir(determinant_folder_path) if f.endswith(".txt")]
eigenvalue_file_list = [f for f in os.listdir(eigenvalue_folder_path) if f.endswith(".txt")]

# Randomly select 1000 files from each folder
selected_determinant_files = random.sample(determinant_file_list, 1000)
selected_eigenvalue_files = random.sample(eigenvalue_file_list, 1000)


def process_files(file_list, folder_path):
    """Reads and processes files from a specified folder, extracting problem and answer pairs."""

    processed_data = []

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "r") as f:
            content = f.read()

            # Split the content into problem and answer
            problem, answer = content.split("Answer:\n")

            # Since the problem set contains unnecessary characters, we remove them for prepration of the dataset
            problem = problem.replace("Answer:\n", "")
            problem = re.sub(r"\s+", " ", problem)
            problem = problem.replace("Problem:", "")

            # Assign the question and answer contents to the dictionary
            processed_data.append({"Question": problem, "Answer": answer})

    return processed_data


# Initialize data list
data = []

# Process existing files
data.extend(process_files(selected_determinant_files, determinant_folder_path))
data.extend(process_files(selected_eigenvalue_files, eigenvalue_folder_path))

# Write to JSON
with open("linear_algebra_data_test.json", "w") as f:
    json.dump(data, f, indent=4)