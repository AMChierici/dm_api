import json
from flask import Flask, request
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate
from sqlalchemy_utils import database_exists, create_database
import pandas as pandas
import numpy as np

model_path = "deepset/sentence_bert"
sql_url = "postgresql:///{}?client_encoding=utf8"
faiss_file_path = "faiss_indices/{}"

# db = SQLAlchemy()

app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = sql_url.format("margarita1234")
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
 
# db.init_app(app)
# migrate = Migrate(app, db)


@app.route('/dialogue_manager', methods=['GET'])

def dialogue_manager():
    if request.method == 'GET':
        raw_data = request.get_json()
        query = raw_data['query']
        avatar_id = raw_data['avatar_id']

        avatar_sql_url = sql_url.format(avatar_id)
        avatar_faiss_file_path = faiss_file_path.format(avatar_id)

        if query is None:
            return 'Please enter a query', 400
        
        document_store = FAISSDocumentStore.load(
                faiss_file_path=avatar_faiss_file_path,
                sql_url=avatar_sql_url)

        retriever = EmbeddingRetriever(document_store=document_store, 
                               embedding_model=model_path, 
                               use_gpu=False)

        query_embedding = np.array(
            retriever.embed_queries(texts=query)
        )
        response = document_store.query_by_embedding(
            query_embedding, 
            top_k=1, 
            return_embedding=False
        )
        answer = response[0].meta['answer']
        id_video = response[0].meta['id_video']
        result = {
            'answer': answer,
            'id_video': id_video

        }
        json.dumps(result)
        return result



@app.route('/update_avatar', methods=['POST'])

def update_avatar():
    if request.method == 'POST':

        raw_data = request.get_json()

        avatar_id = raw_data['avatar_id']
        question = raw_data['question']
        answer = raw_data['answer']
        id_video = raw_data['id_video']

        avatar_sql_url = sql_url.format(avatar_id)
        avatar_faiss_file_path = faiss_file_path.format(avatar_id)

        if not database_exists(avatar_sql_url):
            create_database(avatar_sql_url)
            document_store = FAISSDocumentStore(sql_url=avatar_sql_url)
        else:
            document_store = FAISSDocumentStore.load(
                faiss_file_path=avatar_faiss_file_path,
                sql_url=avatar_sql_url)
        
        retriever = EmbeddingRetriever(document_store=document_store, 
                               embedding_model=model_path, 
                               use_gpu=False)
        docs_to_index = [{
            'text': question,
            'answer': answer,
            'id_video': id_video
        }]

        document_store.write_documents(docs_to_index)
        document_store.update_embeddings(
            retriever=retriever
            )

        document_store.save(faiss_file_path.format(avatar_id))

        return "UPDATED {}".format(avatar_id)


if __name__ == '__main__':
    app.run()