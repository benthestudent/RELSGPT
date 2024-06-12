from openai import OpenAI
import os
import tiktoken
import pandas as pd
import sys
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from google.cloud.firestore_v1.vector import Vector
# Use a service account. REPLACE WITH CREDENTIALS FILE
cred = credentials.Certificate('seniorproj-4d9c3-3e8ebcabc4ea.json')

# Application Default credentials are automatically created.
app = firebase_admin.initialize_app(cred)
db = firestore.client()

client = OpenAI()
MAX_TOKENS = 1600
MODEL = "gpt-3.5-turbo"
BATCH_SIZE = 1000

EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def clean_doc(content):
    # remove leading or trailing whitespace
    metadata, text = content
    text.strip()

    return (metadata, text)

def keep(content):
    # get rid of small documents
    metadata, text = content
    if len(text) < 20:
        return False
    return True

def tokens_per_string(text, model=MODEL):
    # get number of tokens needed for a string
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_text(string, delimiter):
    # split the string on the delimiter provided. If there are 1 or 2 chunks, return
    chunks = string.split(delimiter)
    if(len(chunks)) == 1:
        return [string, ""]
    if len(chunks) == 2:
        return chunks
    
    # If there are more than two chunks, find delimiter nearest to the halfway point and divide there
    num_tokens = tokens_per_string(string)
    halfway = num_tokens // 2
    best_diff = halfway

    for i, chunk in enumerate(chunks):
        left = delimiter.join(chunks[:i + 1])
        left_tokens = tokens_per_string(left)
        diff = abs(halfway-left_tokens)
        # if the current left most chunk(s) does not make up at least half, keep adding chunks together
        if diff >= best_diff:
            break
        best_diff = diff
    left = delimiter.join(chunks[:i])
    right = delimiter.join(chunks[i:])
    return [left, right]

def trunstr(string, model=MODEL, max_tokens=MAX_TOKENS):
    # shorten a string so it is leq max tokens
    encoding = tiktoken.encoding_for_model(model)
    encoded_str = encoding.encode(string)
    trun_str = encoding.decode(encoded_str[:max_tokens])
    if len(encoded_str) > max_tokens:
        print(f"Truncated string from {len(encoded_str)} tokens to {max_tokens} tokens")
    return trun_str

def split_to_chunks(content, model=MODEL, max_tokens = MAX_TOKENS, max_recursion = 5):
    # split a doc of text into a list of chunks in which each chunk has less than
    # max_tokens number of tokens.

    token_len = tokens_per_string(content)

    if token_len <= max_tokens:
        return [content]
    if max_recursion == 0:
        return [trunstr(content, model, max_tokens)]
    
    for delimiter in ["\n\n", "\n", ". "]:
        l, r = split_text(content, delimiter)
        if l == "" or r == "":
            continue # try again with a different delimiter
        else:
            # recursively split each half until they are of the desired size
            results = []
            for half in [l,r]:
                half_strings = split_to_chunks(half, model, max_tokens, max_recursion)
                results.extend(half_strings)
            return results
    # if a split could not be found, then truncate the string
    return [trunstr(content, model, max_tokens)]

    
def read_doc(source, path, link, topic):
    doc_link = link
    # attempt to open document
    print(path)
    with open(path, "r") as f:
        # check if the link is in the first line of the document
        if link == "In-text":
            doc_link = f.readline()
        content = f.read()
    
    metadata = {"source": source,
                "link": doc_link,
                "topic": topic,
                "path": path.split("/")[-1]}
    return (metadata, content)

def read_source(source, path, link, subdir, topic):
    source_docs = [] # list of documents gathered from source
    # go through every document in path, collect metadata, and read text
    # check to see if path exists
    if os.path.isdir(path):
        # a source directory may contain subdirectories for each topic
        # if subdir and pathname are true, then get list of directories
        # otherwise, iterate through files
        if subdir == True and topic == "Pathname":
            # get a list of directories and use recursion to read each document
            for entry in os.scandir(path):
                if entry.is_dir():
                    source_docs.extend(read_source(source, path + "/" + entry.name, link, False, entry.name))
                doc_chunks = []
                
                for doc in source_docs:
                    metadata, text = doc
                    print("Splitting doc " + metadata["path"])
                    chunks = split_to_chunks(text)
                    doc_chunks.append((metadata, chunks))
                embed_docs(doc_chunks)
                source_docs = []
        else:
            # get a list files in directory, and read each one
            for entry in os.scandir(path):
                if entry.is_file() and (entry.name[-4:] == '.txt'):
                    filepath = path + "/" + entry.name
                    if(topic == "Pathname"):
                        new_topic = entry.name.replace(".txt","")
                        new_doc = read_doc(source, filepath, link, new_topic)

                    else:
                        new_doc = read_doc(source, filepath, link, topic)
                    if (len(new_doc[1]) > 0):
                        source_docs.append(new_doc)
    return source_docs


def parse_docs(sources):
    # Go through list of document sources and gather info and content
    sources = pd.read_csv(sources)
    docs = []
    for index, row in sources.iterrows():
        source =  row['Source Name']
        path = row['Path to Docs']
        link = row['Link']
        subdir = True if row['Subdirectories'] == "Yes" else False
        topic = row['Topic']
        docs.extend(read_source(source, path, link, subdir, topic))
        # Split document into chuncks
        doc_chunks = []
        # print("splitting the docs")
        # if not subdir:
        #     for doc in docs:
        #         metadata, text = doc
        #         chunks = split_to_chunks(text)
        #         doc_chunks.append((metadata, chunks))
        #     embed_docs(doc_chunks)
        docs = []

    return doc_chunks

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def embed_docs(doc_chunks):

    for item in doc_chunks:
        # Generate text embeddings
        texts = []
        embeddings = []
        metadata, strings = item
        for batch_start in range(0, len(strings), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = strings[batch_start:batch_end]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            for i, be in enumerate(response.data):
                assert i == be.index # make sure embeddings are in same order as input
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        # for each text and embedding, save a document with metadata
        for i in range(0, len(embeddings)):
            doc = {
                "source": metadata["source"],
                "link": metadata["link"],
                "topic": metadata["topic"],
                "text": strings[i],
                "embedding": Vector(embeddings[i]),
                "date_added": firestore.SERVER_TIMESTAMP,
                "size": len(strings[i])
            }
            # add doc to collection
            try:
                num = "" if i == 0 else "-" + str(i+1)
                name = metadata["source"] + "-" + metadata["topic"] + metadata["path"][:-4] + num
                doc_ref = db.collection("knowledge_base").document(name)
                doc_ref.set(doc)
                print("added {} to knowledge base".format(name))
            except:
                return "Error", 400



# For debugging purposes, save embeddings to csv
# SAVE_PATH = "./embeddings.csv"
# df.to_csv(SAVE_PATH, index=False)

def upload(df):
    for index, row in df.iterrows():
        source =  row['source']
        path = row['link']
        topic = row['topic']
        text = row['text']
        embedding = row['embedding']



def main():

    # open file and parse documents into chunks
    sources = "sources.csv"
    doc_chunks = parse_docs(sources)
    

main()
