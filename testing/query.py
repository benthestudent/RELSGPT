from openai import OpenAI
import os
import tiktoken
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
import random
# Use a service account. REPLACE WITH CREDENTIALS FILE
cred = credentials.Certificate('seniorproj-4d9c3-3e8ebcabc4ea.json')
# Application Default credentials are automatically created.
app = firebase_admin.initialize_app(cred)
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
    
    intro = 'Use the below texts from religious studies sources to answer the subsequent question. ' \
    'If the answer cannot be found in the articles, write "I could not find an answer."'
    question = "\nQuestion: " + query
    msg = intro
    texts = []
    i = 1
    for doc in docs:
        source = "\n\nConsider the following source (ID: {}) on the topic of {}: {}".format(i, doc.get("topic"), doc.get("text"))
        i += 1
        if(num_tokens(msg + source + question, model) > budget):
            break
        else:
            msg += source

    return msg + question

def ask(query, kb, model = MODEL, budget = 4096 - 500):
    msg = gen_query(query, kb, model, budget)
    print("generated message:")
    print(msg)
    messages = [
        {"role": "system", "content": "You answer questions about religion and religious studies."},
        {"role": "user", "content": msg},
    ]
    res = client.chat.completions.create(model=model, messages=messages, temperature=0)
    res_msg = res.choices[0].message.content
    return res_msg

def ask_without_retrieval(query, model=MODEL, budget = 4096-500):
    messages = [
        {"role": "system", "content": "You answer questions about religion and religious studies."},
        {"role": "user", "content": query},
    ]
    res = client.chat.completions.create(model=model, messages=messages, temperature=0)
    res_msg = res.choices[0].message.content
    return res_msg


# while(True):
#     question = input("question: ")
#     answer = ask(question, collection)
#     #answer = ask_without_retrieval(question)
#     print("RELSGPT: " + answer)


# takes an input file that is a list of prompts, queries the test model and a standard GPT model
# outputs responses into out_file, labled with unique key
# also stores file of keys
def test(in_file, out_file):
    used_IDs = []
    keys = []
    with open(in_file, 'r') as f:
        query = f.readline()
        
        while query != '':
            #generate IDs
            test_ID = random.randint(1, 100)
            control_ID = random.randint(1, 100)
            while control_ID == test_ID or test_ID in used_IDs or control_ID in used_IDs:
                control_ID = random.randint(1, 100)
                test_ID = random.randint(1, 100)
            used_IDs.append(control_ID)
            used_IDs.append(test_ID)
            keys.append({"key": control_ID,
                         "model": "control"})
            keys.append({"key": test_ID,
            "model": "test"})
            # ask queries
            test_answer = ask(query, collection)
            control_answer = ask_without_retrieval(query)

            #write to file
            with open(out_file, 'a') as out:
                if(random.randint(1,100) > 50):
                    out.write("{} - {}\n".format(control_ID, control_answer))
                    out.write("{} - {}\n".format(test_ID, test_answer))
                else:
                    out.write("{} - {}\n".format(test_ID, test_answer))
                    out.write("{} - {}\n".format(control_ID, control_answer))
            query = f.readline()
    with open('keys', 'w') as keyfile:
        for key in keys:
            keyfile.write("{} - {}\n".format(key["key"], key["model"]))


    
test('tests', 'answers')                





