import pandas as pd

df = pd.read_excel('data/Test Files.xlsx')

def generate_email(name):
    # Remove special characters and spaces from the name
    name = ''.join(e for e in name if e.isalnum() or e.isspace())

    # Split the name into parts
    name_parts = name.split()

    # Ensure there is at least one name part
    if len(name_parts) >= 1:
        # Get the first part as the first name
        first_name = name_parts[0]

        # If there is more than one part, use the last part as the last name
        last_name = name_parts[-1] if len(name_parts) > 1 else ''

        # Create the email address by combining the first initial and last name
        email = f"{first_name[0]}{last_name}".lower() + "@gmail.com"

        # Ensure email address uniqueness
        counter = 1
        original_email = email
        while email in df['Email Address']:
            email = f"{first_name[0]}{last_name}{counter}".lower() + "@gmail.com"
            counter += 1

        return email
    else:
        # Handle the case when the name doesn't have any parts
        return None



def get_male_students(df):
    return df[df['Gender'] == 'M']['Student Name'].tolist()

def get_female_students(df):
    return df[df['Gender'] == 'F']['Student Name'].tolist()




