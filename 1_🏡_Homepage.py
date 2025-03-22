import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import os
import re
import json
from dotenv import load_dotenv

# Load Gemini API Key
load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found! Check your .env file.")
genai.configure(api_key="")

# Title and UI
st.set_page_config(page_title="GradeGraph 2.0", page_icon="📊")
st.markdown('<div style="text-align: center;"><h1>📊 GradeGraph 2.0</h1></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="text-align: center;"><span style="font-size: 10.5rem;">📊</span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="text-align: center;"><span style="font-size: 2.5rem;">GradeGraph 2.0</span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="text-align: center;"><span style="font-size: 1.5rem;">"Developed by Data Science Students"</span></div>', unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_student_data(text):
    prompt = f"""
    Extract student details from the following exam result sheet:

    {text}

    - Identify the **Seat No** first, then extract all details from that same line.
    - The **Result** field should always be either "P" (Pass) or "F" (Fail).
      - Convert "P#" to "P", "PF" to "P", and "FF" to "F".
    - Return **only JSON**, formatted like this:

    ```json
    [
        {{
            "Seat No": "CDS621101",
            "Student Name": "John Doe",
            "Total Marks": "468",
            "SGPI": "6.83",
            "CGPI": "6.3",
            "Result": "P"
        }}
    ]
    ```
    
    - **Do NOT include explanations**—just return pure JSON.
    - Ensure there are **no missing brackets, quotes, or extra text**.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    raw_text = response.candidates[0].content.parts[0].text  # Extract raw JSON text

    # Remove Markdown-style JSON block if present
    cleaned_json_text = re.sub(r"```json\n|\n```", "", raw_text).strip()

    try:
        return json.loads(cleaned_json_text)  # Convert to structured JSON
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}. Raw Response:\n{cleaned_json_text}")
        return None


# File Upload UI
uploaded_pdf_file = st.file_uploader("Upload LTCE Result PDF", type=["pdf"])

if uploaded_pdf_file is not None:
    st.write("Extracting and processing data...")

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(uploaded_pdf_file)

    # Process via Gemini
    student_data = extract_student_data(pdf_text)

    if student_data:
        # Convert extracted data into DataFrame
        df = pd.DataFrame(student_data)

        # Display cleaned data
        st.write(df)

        # Proceed to visualization
        st.subheader("Results Visualization")

        # Total students, pass/fail count
        total_students = len(df)
        pass_count = df[df['Result'].str.contains('P', case=False, na=False)].shape[0]
        fail_count = total_students - pass_count

        st.markdown(f"**Total Students:** {total_students}")
        st.markdown(f'<div style="color:green;">Passed: {pass_count}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:red;">Failed: {fail_count}</div>', unsafe_allow_html=True)

        # Data visualization
        student_names = df['Student Name'].unique().tolist()
        student_names.sort()
        selected_student = st.selectbox("Select a student or 'Overall'", ["Overall"] + student_names)

        if selected_student != "Overall":
            student_df = df[df['Student Name'] == selected_student]
            
            if not student_df.empty:
                st.markdown(f"**CGPI:** {student_df['CGPI'].values[0]}")
                st.markdown(f"**SGPI:** {student_df['SGPI'].values[0]}")
                
                pass_status = student_df['Result'].values[0]
                st.markdown(
                    f"<div style='background-color: {'limegreen' if pass_status == 'P' else 'tomato'}; "
                    "padding: 10px; border-radius: 5px;'>"
                    f"{'🎉 Passed' if pass_status == 'P' else '😓 Failed'}</div>", 
                    unsafe_allow_html=True
                )

                # Ranks
                df_sorted = df.sort_values(by='CGPI', ascending=False)
                df_sorted['Rank'] = range(1, len(df_sorted) + 1)
                student_rank = df_sorted[df_sorted['Student Name'] == selected_student]['Rank'].values[0]

                st.markdown(f"**Overall Rank:** {student_rank}")

        else:
            # Pass/Fail Pie Chart
            st.subheader("Pass/Fail Distribution")
            pass_fail_counts = df['Result'].value_counts()
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(pass_fail_counts, labels=pass_fail_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Top 5 Students by CGPI
            st.subheader("Top 5 Students (CGPI)")
            top_students = df.sort_values(by='CGPI', ascending=False).head(5)
            sns.barplot(data=top_students, x='Student Name', y='CGPI')
            st.pyplot()

        # Footer
        st.markdown("---")
        st.markdown('<div style="text-align: center;">© 2025 GradeGraph</div>', unsafe_allow_html=True)

