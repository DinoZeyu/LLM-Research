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

# Initialize an empty list to store the data
data = []

# Read and process determinant files
for file_name in selected_determinant_files:
  file_path = os.path.join(determinant_folder_path, file_name)
  with open(file_path, "r") as f:
    content = f.read()
    problem, answer = content.split("Answer:\n")
    problem = problem.replace("Answer:\n", "")
    data.append({"Question": content, 'Answer': answer })  

# Read and process eigenvalue files
for file_name in selected_eigenvalue_files:
  file_path = os.path.join(eigenvalue_folder_path, file_name)
  with open(file_path, "r") as f:
    content = f.read()
    problem, answer = content.split("Answer:\n")
    problem = problem.replace("Answer:\n", "")
    data.append({"Question": content, "Answer": answer})  

# Create a JSON file and write the data
with open("linear_algebra_data.json", "w") as f:
  json.dump(data, f, indent=4)