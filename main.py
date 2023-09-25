import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from transformers import AutoModel
import json
import os
import torch
import random
import re


model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

from functions import get_male_students, get_female_students, generate_email

logs = 'logs'
log_file_path = os.path.join(logs, 'app.log')

df_3b = pd.read_excel('data/Test Files.xlsx', sheet_name='3B')
df_3c = pd.read_excel('data/Test Files.xlsx', sheet_name='3C')

combined_df = pd.concat([df_3b, df_3c], ignore_index=True)

# Extract male and female names from the DataFrame
male_names = combined_df[combined_df['Gender'] == 'M']['Student Name'].tolist()
female_names = combined_df[combined_df['Gender'] == 'F']['Student Name'].tolist()

# Apply the generate_email function to create a new 'Email Address' column
combined_df['Email Address'] = combined_df['Student Name'].apply(lambda name: generate_email(name, combined_df))

tsv_file_path = 'data/student_data.tsv'
csv_file_path = 'data/student_data.csv'

# Save the data as TSV (Tab-Separated Values)
combined_df.to_csv(tsv_file_path, sep='\t', index=False)

# Save the data as CSV (Comma-Separated Values)
combined_df.to_csv(csv_file_path, index=False)

# Filter and separate students with special characters
special_character_names = combined_df[combined_df['Student Name'].str.contains(r'[^\w\s\\\'-]', regex=True, na=False)]
special_character_names_list = special_character_names['Student Name'].tolist()

# Separate Male and Female students
male_students = get_male_students(combined_df)
female_students = get_female_students(combined_df)

# Log the number of Male and Female students
num_male_students = len(male_students)
num_female_students = len(female_students)

with open(log_file_path, 'w') as log_file:
    log_file.write(f'Number of Male Students: {num_male_students}\n')
    log_file.write(f'Number of Female Students: {num_female_students}\n')

    # Log the names of students with special characters
    log_file.write('Names of Students with Special Characters:\n')
    for name in special_character_names_list:
        log_file.write(f'{name}\n')

print(f'Number of Male Students: {num_male_students}')
print(f'Number of Female Students: {num_female_students}')


# Merge all the shuffled names and save as a JSON file
all_names = male_names + female_names
random.shuffle(all_names)
shuffled_df = pd.DataFrame({'Shuffled Names': all_names})
shuffled_json_file = 'data/shuffled_names.json'
shuffled_df.to_json(shuffled_json_file, orient='split')
print(f"Merged and shuffled {len(all_names)} names saved to '{shuffled_json_file}'")
