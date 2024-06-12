from flask import Flask, request, session
from flask_cors import CORS, cross_origin
from openai import OpenAI
import os
import tiktoken
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

app = Flask(__name__)
cors = CORS(app)
# Use a service account. REPLACE WITH CREDENTIALS FILE
cred = credentials.Certificate('seniorproj-4d9c3-3e8ebcabc4ea.json')
# Application Default credentials are automatically created.
default = firebase_admin.initialize_app(cred)
db = firestore.client()

MAX_TOKENS = 1600
MODEL = "gpt-3.5-turbo"
BATCH_SIZE = 1000

EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
collection = db.collection("knowledge_base")

def num_tokens(text, model=MODEL):
    # get number of tokens needed for a string
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def embed(content):
    batch = [content]
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response.data):
        assert i == be.index  # make sure embeddings are in same order as input
    batch_embeddings = [e.embedding for e in response.data]
    emb = batch_embeddings[0]
    return emb

# generate a query with relevant information to send to GPT
def gen_query(query, kb, model, budget):
    embedded_q = embed(query)
    # use vector search to find related documents
    docs = (collection.find_nearest(
    vector_field="embedding",
    query_vector=Vector(embedded_q),
    distance_measure=DistanceMeasure.EUCLIDEAN,
    limit=2).stream())
    
    intro = 'Use the below texts from religious studies sources to answer the subsequent question. Each source has a numerical id indicated by "ID: ". Whenever you use a detail or piece of information from a source, write the source\'s ID number in brackets at the end of the sentence like this [x] where x is the source\'s ID number. ' \
    'If the answer cannot be found in the articles, write "I could not find an answer."'
    question = "\n\nQuestion: " + query
    msg = intro
    texts = []
    sources = []
    i = 1
    for doc in docs:
        source = "\n\nConsider the following source (ID: {}) on the topic of {}: {}".format(i, doc.get("topic"), doc.get("text"))
        i += 1
        if(num_tokens(msg + source + question, model) > budget):
            break
        else:
            msg += source
            sources.append({"source": doc.get("source"), "link": doc.get("link"), "topic": doc.get("topic")})

    return (msg + question, sources)

def ask(query, kb, model = MODEL, budget = 4096 - 500):
    msg, sources = gen_query(query, kb, model, budget)
    messages = [
        {"role": "system", "content": "You answer questions about religion and religious studies."},
        {"role": "user", "content": msg},
    ]
    res = client.chat.completions.create(model=model, messages=messages, temperature=0)
    res_msg = res.choices[0].message.content
    if "I could not find an answer" in res_msg :
        return res_msg
    citation = "<br>Sources:"
    i = 1
    for source in sources:
        citation += "<p><a href='{}'>[{}] {}-{}</a></p>".format(source["link"], i, source["source"], source["topic"])
        i += 1
    return res_msg + citation



@app.route('/ask', methods=['GET'])
def query():

    args = request.args
    question = args.get("question")
    if not question:
        return "Error: no question provided", 400
    
    answer = ask(question, collection)
    print(answer)

    return answer # change later

if __name__ == '__main__':
    app.run()

