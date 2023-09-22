import pandas as pd
import os



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



