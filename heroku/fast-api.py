from fastapi import FastAPI, HTTPException
import requests
import PyPDF2
import tempfile
import os
import subprocess
import openai
from pydantic import BaseModel
from scipy.spatial import distance
import numpy as np
import tiktoken 
from scipy import spatial
import pandas as pd
import ast

app = FastAPI()


EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
embeddings = []
df = pd.DataFrame(columns=["text", "embedding"])
openai.api_key = os.environ["OPENAI_API_KEY"]

def download_pdf(pdf_url, file_name):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(file_name, "wb") as pdf_file:
            pdf_file.write(response.content)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {e}")
        


def pypdf_extract(file_name):
    try:
        text = ''
        with open(file_name, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text += page.extract_text()
        return text
    except PyPDF2.utils.PdfReadError as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")
    
    
def embed(text):
    data = [
    {    
        'heading': "Full Content",
        'content': text.strip(),
    }
    ]
    df = pd.DataFrame(data)
    # embeddings = embed(df['content'])
    text_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    print("Here")

    text_embedding = text_embedding_response["data"][0]["embedding"]
    print(len(text_embedding))
    result_df = pd.DataFrame({'text': df['content'], 'embedding': [text_embedding]})
    result_df.to_csv("output_with_embeddings.csv", index=False)
    print("CSV created")
    


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
    ) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    query_embedding = query_embedding_response["data"][0]["embedding"]
    # Ensure query embedding has the same dimension as the DataFrame embeddings
    query_embedding = np.array(query_embedding)
    print("in relatedness")
    embeddings= df["embedding"][0]
    print(len(embeddings))
    embeddings= np.array(embeddings)
    
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, embeddings))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]




def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
    ) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    # print("in query msg"+str(type(df)))
    # print(df)
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below content of Eligibility Requirements to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nEligibility Requirements:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question

def ask(
    query: str,
    df: pd.DataFrame=df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    # print("in ask"+str(type(df)))
    df = pd.read_csv("output_with_embeddings.csv")
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Financial Statements for Tier 1 Off erings."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

  
class PDFRequest(BaseModel):
    pdf_url: str

class n_pdf(BaseModel):
    text: str

class PDFResponse(BaseModel):
    data: str

@app.post("/process_pdf_pypdf")
async def process_pdf(pdf_data: PDFRequest):
    global df
    pdf_url = pdf_data.pdf_url


    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
            temp_file_name = temp_pdf_file.name

        download_pdf(pdf_url, temp_file_name)
        try:
            extracted_text = pypdf_extract(temp_file_name)
        except:
            raise HTTPException(status_code=400, detail="Invalid extraction method. Use 'pypdf'")
        
        
        embed(extracted_text)
        return PDFResponse(data="success")
    finally:
        if temp_file_name:
            try:
                os.remove(temp_file_name)
            except Exception as e:
                pass


@app.post("/process_pdf_nougat")
async def process_pdf(pdf_data: n_pdf):
    global df
    extracted_text = pdf_data.text
        
    try:
        embed(extracted_text)
        return PDFResponse(data="success")

    except:
        pass


@app.get("/get_answer")
async def get_answer(question: str):
    response = ask(question, embeddings)
    return {"answer": response}