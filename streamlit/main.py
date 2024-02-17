import requests
import streamlit as st
import time
import os
import subprocess
import PyPDF2


st.title("API Requests")
pdf_link = st.text_input("Enter PDF Link:")
method = st.selectbox("Select Method:", ["PyPdf", "Nougat"])
post_button = st.button("Analyze")
question = st.text_input("Enter a Question:")
get_button = st.button("Answers")


# Function to make the API request for getting an answer
def make_question_api_request(question):
    api_url = f"https://testing-assignment-2-b10953b0ae68.herokuapp.com/get_answer?question={question}"
    headers = {
        "accept": "application/json",
    }

    response = requests.get(api_url, headers=headers)

    return response



def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

def nougat_extract(pdf_path):
    try:
        path = pdf_path
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        with st.spinner("Processing..."):
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Use CPU if GPU not available
            subprocess.run(['nougat', '--markdown', 'pdf', path, '--out', '.'], check=True)
            st.success("Process completed!")
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            file_contents = f'{pdf_name}.mmd'
            file = open(file_contents, 'r')
            return file.read()
    
    except Exception as e:
        st.error(f"Error during Nougat extraction: {e}")
        return ""
    
def process_pdf_pypdf(pdf_link):
    try:
    # Call your FastAPI endpoint for processing PDF with PyPDF2 here
        endpoint_url = "https://testing-assignment-2-b10953b0ae68.herokuapp.com/process_pdf_pypdf"
        data = {"pdf_url": pdf_link}
        response = requests.post(endpoint_url, json=data)
        st.success("PDF processing with PyPDF2 completed!")
    except Exception as e:
        st.error(f"Error during PyPDF2 extraction: {e}")

# Function to process extracted PDF text using Nougat
def process_pdf_nougat(pdf_text):
    try:
    # Call your FastAPI endpoint for processing extracted PDF text with Nougat here
        endpoint_url = "https://testing-assignment-2-b10953b0ae68.herokuapp.com/process_pdf_nougat"
        data = {"text": pdf_text}
        response = requests.post(endpoint_url, json=data)
        st.success("PDF text processing with Nougat completed!")
    except Exception as e:
        st.error(f"Error during Nougat extraction: {e}")

# Handle button clicks to make the API requests
if post_button:
    if pdf_link and method:
        file_name = pdf_link.split("/")[-1]
        download_pdf(pdf_link, file_name)
        if method == 'PyPdf':
            process_pdf_pypdf(pdf_link)


        elif method == 'Nougat':
            text = nougat_extract(file_name)
            process_pdf_nougat(text)


elif get_button:
    if question:
        response = make_question_api_request(question)
        if response.status_code == 200:
            st.success("GET API Request for Question Successful")
            st.json(response.json())
        else:
            st.error("GET API Request for Question Failed")
            st.text(response.text)
    else:
        st.warning("Please enter a question for GET request.")


