from openai import OpenAI
from langsmith.wrappers import wrap_openai  # traces openai calls
from dotenv import load_dotenv


load_dotenv()

def retriever(query: str):
    return ["Harrison worked at Kensho"]

client = wrap_openai(OpenAI())  # log traces by wrapping the model calls

def rag(question: str) -> str:
    docs = retriever(question)
    system_message = (
        "Answer the user's question using only the provided information below:\n"
        + "\n".join(docs)
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print(rag("Where did Harrison work?"))