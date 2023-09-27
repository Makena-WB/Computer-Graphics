import pandas as pd
import os
import json
from transformers import LaBSE, LaBSETokenizer



from functions import get_male_students, get_female_students, generate_email

logs = 'logs'
log_file_path = os.path.join(logs, 'app.log')


df = pd.read_excel('data/Test Files.xlsx')

# Extract male and female names from the DataFrame
male_names = df[df['Gender'] == 'M']['Student Name'].tolist()
female_names = df[df['Gender'] == 'F']['Student Name'].tolist()

# Load the student data from the Excel file

# Apply the generate_email function to create a new 'Email Address' column
df['Email Address'] = df['Student Name'].apply(generate_email)

tsv_file_path = 'data/student_data.tsv'
csv_file_path = 'data/student_data.csv'

# Save the data as TSV (Tab-Separated Values)
df.to_csv(tsv_file_path, sep='\t', index=False)

# Save the data as CSV (Comma-Separated Values)
df.to_csv(csv_file_path, index=False)



male_students = get_male_students(df)
female_students = get_female_students(df)

# Log the number of male and female students
num_male_students = len(male_students)
num_female_students = len(female_students)

with open(log_file_path, 'w') as log_file:
    log_file.write(f'Number of Male Students: {num_male_students}\n')
    log_file.write(f'Number of Female Students: {num_female_students}\n')

# List names of students with special characters using regex (if needed)
special_character_names = df[df['Student Name'].str.contains(r'[^\w\s\\\'-]', regex=True, na=False)]
special_character_names_list = special_character_names['Student Name'].tolist()

# Log the names of students with special characters to the same log file
with open(log_file_path, 'a') as log_file:
    log_file.write('Names of Students with Special Characters:\n')
    for name in special_character_names_list:
        log_file.write(f'{name}\n')

print(f'Number of Male Students: {num_male_students}')
print(f'Number of Female Students: {num_female_students}')


# Load male and female names from TSV and CSV files
male_names = pd.read_csv('male_names.csv')['name'].tolist()
female_names = pd.read_csv('female_names.tsv', sep='\t')['name'].tolist()

# Preprocess names
tokenizer = LaBSETokenizer.from_pretrained("setu4993/LaBSE")
male_names = [tokenizer.tokenize(name.lower()) for name in male_names]
female_names = [tokenizer.tokenize(name.lower()) for name in female_names]

# Load LaBSE model
model = LaBSE.from_pretrained("setu4993/LaBSE")

# Calculate similarity scores
similarity_matrix = []
for male_name in male_names:
    row = []
    for female_name in female_names:
        inputs = tokenizer.encode_plus(
            male_name, female_name, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            embeddings = model(**inputs).pooler_output
        similarity_score = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
        row.append(similarity_score.item())
    similarity_matrix.append(row)

# Filter results with at least 50% similarity
filtered_results = []
for i, male_name in enumerate(male_names):
    for j, female_name in enumerate(female_names):
        if similarity_matrix[i][j] >= 0.5:
            filtered_results.append({
                "male_name": male_name,
                "female_name": female_name,
                "similarity_score": similarity_matrix[i][j]
            })

# Save filtered results in a JSON file
with open('similarity_results.json', 'w') as f:
    json.dump(filtered_results, f)









