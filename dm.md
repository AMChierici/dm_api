Run only for the first time:


```python
# Install the latest release of Haystack in your own environment 
#!pip install git+https://github.com/deepset-ai/haystack.git

# If running on GPUs, e.g., DALMA
# Install the latest master of Haystack
#!pip install git+https://github.com/deepset-ai/haystack.git
#!pip install urllib3==1.25.4
#!pip install torch==1.6.0+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```


```python
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers
import pandas as pd
import numpy as np
import pickle

```

    06/13/2021 09:57:18 - INFO - faiss -   Loading faiss.



```python
# FAISS Document Store

document_store = FAISSDocumentStore(
    sql_url="postgresql:///margarita1234?client_encoding=utf8"
)
```


```python
model_path = "deepset/sentence_bert"

retriever = EmbeddingRetriever(document_store=document_store, 
                               embedding_model=model_path, 
                               use_gpu=False)
```

    06/13/2021 09:57:22 - INFO - haystack.retriever.dense -   Init retriever using embeddings of model deepset/sentence_bert
    06/13/2021 09:57:22 - INFO - farm.utils -   Using device: CPU 
    06/13/2021 09:57:22 - INFO - farm.utils -   Number of GPUs: 0
    06/13/2021 09:57:22 - INFO - farm.utils -   Distributed Training: False
    06/13/2021 09:57:22 - INFO - farm.utils -   Automatic Mixed Precision: None
    06/13/2021 09:57:33 - WARNING - farm.utils -   ML Logging is turned off. No parameters, metrics or artifacts will be logged to MLFlow.
    06/13/2021 09:57:33 - INFO - farm.utils -   Using device: CPU 
    06/13/2021 09:57:33 - INFO - farm.utils -   Number of GPUs: 0
    06/13/2021 09:57:33 - INFO - farm.utils -   Distributed Training: False
    06/13/2021 09:57:33 - INFO - farm.utils -   Automatic Mixed Precision: None
    06/13/2021 09:57:33 - WARNING - haystack.retriever.dense -   You seem to be using a Sentence Transformer with the dot_product function. We recommend using cosine instead. This can be set when initializing the DocumentStore



```python
# Get dataframe with columns "question", "answer" and some custom metadata
df = pd.read_csv("data/MargaritaCorpusKB_video_id.csv", encoding='utf-8')
df = df[["Context", "Utterance", "id_video"]]
df = df.rename(columns={"Context": "text", "Utterance": "answer"})
df.drop_duplicates(subset=['text'], inplace=True)
df.drop_duplicates(subset=['answer'], inplace=True)
# Minimal cleaning
df.fillna(value="", inplace=True)
df["text"] = df["text"].apply(lambda x: x.strip())
# Drop question that only have *
index_drop = df[df["text"] == "*"].index
df.drop(index_drop, inplace=True)

# Get embeddings for our questions from the FAQs
# questions = list(df["text"].values)
# df["embedding"] = retriever.embed_queries(texts=questions)

# Convert Dataframe to list of dicts and index them in our DocumentStore
docs_to_index = df.to_dict(orient="records")

# # Delete existing documents in documents store
document_store.delete_all_documents()

# Write documents to document store
document_store.write_documents(docs_to_index)

# Add documents embeddings to index
document_store.update_embeddings(
    retriever=retriever
)
```

    02/11/2021 13:55:09 - INFO - haystack.document_store.faiss -   Updating embeddings for 349 docs...
      0%|          | 0/349 [00:00<?, ?it/s]02/11/2021 13:55:09 - WARNING - farm.data_handler.processor -   Currently no support in InferenceProcessor for returning problematic ids
    
    Inferencing Samples:   0%|          | 0/88 [00:00<?, ? Batches/s][A
    Inferencing Samples:   1%|          | 1/88 [00:04<06:39,  4.59s/ Batches][A
    Inferencing Samples:   2%|▏         | 2/88 [00:09<06:28,  4.52s/ Batches][A
    Inferencing Samples:   3%|▎         | 3/88 [00:13<06:24,  4.53s/ Batches][A
    Inferencing Samples:   5%|▍         | 4/88 [00:18<06:21,  4.54s/ Batches][A
    Inferencing Samples:   6%|▌         | 5/88 [00:22<06:19,  4.57s/ Batches][A
    Inferencing Samples:   7%|▋         | 6/88 [00:27<06:16,  4.59s/ Batches][A
    Inferencing Samples:   8%|▊         | 7/88 [00:32<06:12,  4.60s/ Batches][A
    Inferencing Samples:   9%|▉         | 8/88 [00:36<06:08,  4.60s/ Batches][A
    Inferencing Samples:  10%|█         | 9/88 [00:41<06:05,  4.62s/ Batches][A
    Inferencing Samples:  11%|█▏        | 10/88 [00:45<06:01,  4.64s/ Batches][A
    Inferencing Samples:  12%|█▎        | 11/88 [00:50<05:57,  4.65s/ Batches][A
    Inferencing Samples:  14%|█▎        | 12/88 [00:55<05:54,  4.66s/ Batches][A
    Inferencing Samples:  15%|█▍        | 13/88 [01:00<05:50,  4.68s/ Batches][A
    Inferencing Samples:  16%|█▌        | 14/88 [01:04<05:45,  4.67s/ Batches][A
    Inferencing Samples:  17%|█▋        | 15/88 [01:09<05:40,  4.67s/ Batches][A
    Inferencing Samples:  18%|█▊        | 16/88 [01:13<05:34,  4.65s/ Batches][A
    Inferencing Samples:  19%|█▉        | 17/88 [01:18<05:30,  4.65s/ Batches][A
    Inferencing Samples:  20%|██        | 18/88 [01:23<05:24,  4.63s/ Batches][A
    Inferencing Samples:  22%|██▏       | 19/88 [01:27<05:18,  4.62s/ Batches][A
    Inferencing Samples:  23%|██▎       | 20/88 [01:32<05:15,  4.64s/ Batches][A
    Inferencing Samples:  24%|██▍       | 21/88 [01:37<05:12,  4.67s/ Batches][A
    Inferencing Samples:  25%|██▌       | 22/88 [01:42<05:14,  4.76s/ Batches][A
    Inferencing Samples:  26%|██▌       | 23/88 [01:46<05:07,  4.73s/ Batches][A
    Inferencing Samples:  27%|██▋       | 24/88 [01:51<05:00,  4.69s/ Batches][A
    Inferencing Samples:  28%|██▊       | 25/88 [01:56<04:53,  4.67s/ Batches][A
    Inferencing Samples:  30%|██▉       | 26/88 [02:00<04:47,  4.64s/ Batches][A
    Inferencing Samples:  31%|███       | 27/88 [02:05<04:42,  4.62s/ Batches][A
    Inferencing Samples:  32%|███▏      | 28/88 [02:09<04:37,  4.62s/ Batches][A
    Inferencing Samples:  33%|███▎      | 29/88 [02:14<04:31,  4.60s/ Batches][A
    Inferencing Samples:  34%|███▍      | 30/88 [02:19<04:26,  4.60s/ Batches][A
    Inferencing Samples:  35%|███▌      | 31/88 [02:23<04:21,  4.59s/ Batches][A
    Inferencing Samples:  36%|███▋      | 32/88 [02:28<04:16,  4.59s/ Batches][A
    Inferencing Samples:  38%|███▊      | 33/88 [02:32<04:12,  4.59s/ Batches][A
    Inferencing Samples:  39%|███▊      | 34/88 [02:37<04:08,  4.61s/ Batches][A
    Inferencing Samples:  40%|███▉      | 35/88 [02:42<04:06,  4.65s/ Batches][A
    Inferencing Samples:  41%|████      | 36/88 [02:46<04:02,  4.67s/ Batches][A
    Inferencing Samples:  42%|████▏     | 37/88 [02:51<03:58,  4.67s/ Batches][A
    Inferencing Samples:  43%|████▎     | 38/88 [02:56<03:53,  4.67s/ Batches][A
    Inferencing Samples:  44%|████▍     | 39/88 [03:00<03:49,  4.69s/ Batches][A
    Inferencing Samples:  45%|████▌     | 40/88 [03:05<03:48,  4.75s/ Batches][A
    Inferencing Samples:  47%|████▋     | 41/88 [03:10<03:42,  4.73s/ Batches][A
    Inferencing Samples:  48%|████▊     | 42/88 [03:15<03:35,  4.69s/ Batches][A
    Inferencing Samples:  49%|████▉     | 43/88 [03:19<03:29,  4.66s/ Batches][A
    Inferencing Samples:  50%|█████     | 44/88 [03:24<03:24,  4.64s/ Batches][A
    Inferencing Samples:  51%|█████     | 45/88 [03:28<03:19,  4.64s/ Batches][A
    Inferencing Samples:  52%|█████▏    | 46/88 [03:33<03:15,  4.66s/ Batches][A
    Inferencing Samples:  53%|█████▎    | 47/88 [03:38<03:11,  4.66s/ Batches][A
    Inferencing Samples:  55%|█████▍    | 48/88 [03:42<03:06,  4.65s/ Batches][A
    Inferencing Samples:  56%|█████▌    | 49/88 [03:47<03:00,  4.63s/ Batches][A
    Inferencing Samples:  57%|█████▋    | 50/88 [03:52<02:55,  4.61s/ Batches][A
    Inferencing Samples:  58%|█████▊    | 51/88 [03:56<02:50,  4.60s/ Batches][A
    Inferencing Samples:  59%|█████▉    | 52/88 [04:01<02:47,  4.64s/ Batches][A
    Inferencing Samples:  60%|██████    | 53/88 [04:05<02:41,  4.62s/ Batches][A
    Inferencing Samples:  61%|██████▏   | 54/88 [04:10<02:37,  4.62s/ Batches][A
    Inferencing Samples:  62%|██████▎   | 55/88 [04:15<02:32,  4.61s/ Batches][A
    Inferencing Samples:  64%|██████▎   | 56/88 [04:19<02:27,  4.61s/ Batches][A
    Inferencing Samples:  65%|██████▍   | 57/88 [04:24<02:24,  4.65s/ Batches][A
    Inferencing Samples:  66%|██████▌   | 58/88 [04:29<02:18,  4.60s/ Batches][A
    Inferencing Samples:  67%|██████▋   | 59/88 [04:33<02:13,  4.59s/ Batches][A
    Inferencing Samples:  68%|██████▊   | 60/88 [04:38<02:08,  4.57s/ Batches][A
    Inferencing Samples:  69%|██████▉   | 61/88 [04:42<02:03,  4.57s/ Batches][A
    Inferencing Samples:  70%|███████   | 62/88 [04:47<01:58,  4.56s/ Batches][A
    Inferencing Samples:  72%|███████▏  | 63/88 [04:51<01:53,  4.55s/ Batches][A
    Inferencing Samples:  73%|███████▎  | 64/88 [04:56<01:49,  4.55s/ Batches][A
    Inferencing Samples:  74%|███████▍  | 65/88 [05:01<01:45,  4.59s/ Batches][A
    Inferencing Samples:  75%|███████▌  | 66/88 [05:05<01:42,  4.64s/ Batches][A
    Inferencing Samples:  76%|███████▌  | 67/88 [05:10<01:38,  4.71s/ Batches][A
    Inferencing Samples:  77%|███████▋  | 68/88 [05:15<01:35,  4.77s/ Batches][A
    Inferencing Samples:  78%|███████▊  | 69/88 [05:20<01:30,  4.78s/ Batches][A
    Inferencing Samples:  80%|███████▉  | 70/88 [05:25<01:26,  4.81s/ Batches][A
    Inferencing Samples:  81%|████████  | 71/88 [05:30<01:22,  4.84s/ Batches][A
    Inferencing Samples:  82%|████████▏ | 72/88 [05:34<01:16,  4.81s/ Batches][A
    Inferencing Samples:  83%|████████▎ | 73/88 [05:39<01:11,  4.76s/ Batches][A
    Inferencing Samples:  84%|████████▍ | 74/88 [05:44<01:05,  4.69s/ Batches][A
    Inferencing Samples:  85%|████████▌ | 75/88 [05:48<01:01,  4.75s/ Batches][A
    Inferencing Samples:  86%|████████▋ | 76/88 [05:53<00:56,  4.75s/ Batches][A
    Inferencing Samples:  88%|████████▊ | 77/88 [05:58<00:51,  4.71s/ Batches][A
    Inferencing Samples:  89%|████████▊ | 78/88 [06:02<00:46,  4.68s/ Batches][A
    Inferencing Samples:  90%|████████▉ | 79/88 [06:07<00:41,  4.65s/ Batches][A
    Inferencing Samples:  91%|█████████ | 80/88 [06:11<00:36,  4.61s/ Batches][A
    Inferencing Samples:  92%|█████████▏| 81/88 [06:16<00:32,  4.58s/ Batches][A
    Inferencing Samples:  93%|█████████▎| 82/88 [06:21<00:27,  4.61s/ Batches][A
    Inferencing Samples:  94%|█████████▍| 83/88 [06:25<00:23,  4.61s/ Batches][A
    Inferencing Samples:  95%|█████████▌| 84/88 [06:30<00:18,  4.60s/ Batches][A
    Inferencing Samples:  97%|█████████▋| 85/88 [06:34<00:13,  4.59s/ Batches][A
    Inferencing Samples:  98%|█████████▊| 86/88 [06:39<00:09,  4.62s/ Batches][A
    Inferencing Samples:  99%|█████████▉| 87/88 [06:44<00:04,  4.62s/ Batches][A
    Inferencing Samples: 100%|██████████| 88/88 [06:45<00:00,  4.61s/ Batches][A
    10000it [06:45, 24.65it/s]             



```python
query_embedding = np.array(
    retriever.embed_queries(texts="How are you?")
)

response = document_store.query_by_embedding(
    query_embedding, 
    top_k=1, 
    return_embedding=False
)

print(response[0].meta['answer'])
print(response[0].meta['id_video'])
```

    02/11/2021 14:01:55 - WARNING - farm.data_handler.processor -   Currently no support in InferenceProcessor for returning problematic ids
    Inferencing Samples: 100%|██████████| 1/1 [00:01<00:00,  1.20s/ Batches]

    Pretty good, thank you!
    77157368d489465a3af172497e80ed59


    



```python
document_store.save("faiss_indices/margarita1234")
# outfile = open("faiss_indices/margarita.pkl", 'wb')
# pickle.dump(document_store, outfile)
# outfile.close()
```


```python
# infile = open("faiss_indices/margarita.pkl",'rb')
# new_document_store = pickle.load(infile)

# infile.close()

new_document_store = FAISSDocumentStore.load(faiss_file_path="faiss_indices/margarita1234",
                                             sql_url='postgresql:///margarita1234?client_encoding=utf8')
```


```python
model_path = "deepset/sentence_bert"

retriever = EmbeddingRetriever(document_store=new_document_store, 
                               embedding_model=model_path, 
                               use_gpu=False)
```

    06/13/2021 09:57:40 - INFO - haystack.retriever.dense -   Init retriever using embeddings of model deepset/sentence_bert
    06/13/2021 09:57:40 - INFO - farm.utils -   Using device: CPU 
    06/13/2021 09:57:40 - INFO - farm.utils -   Number of GPUs: 0
    06/13/2021 09:57:40 - INFO - farm.utils -   Distributed Training: False
    06/13/2021 09:57:40 - INFO - farm.utils -   Automatic Mixed Precision: None
    06/13/2021 09:57:51 - WARNING - farm.utils -   ML Logging is turned off. No parameters, metrics or artifacts will be logged to MLFlow.
    06/13/2021 09:57:51 - INFO - farm.utils -   Using device: CPU 
    06/13/2021 09:57:51 - INFO - farm.utils -   Number of GPUs: 0
    06/13/2021 09:57:51 - INFO - farm.utils -   Distributed Training: False
    06/13/2021 09:57:51 - INFO - farm.utils -   Automatic Mixed Precision: None
    06/13/2021 09:57:51 - WARNING - haystack.retriever.dense -   You seem to be using a Sentence Transformer with the dot_product function. We recommend using cosine instead. This can be set when initializing the DocumentStore



```python
query_embedding = np.array(
    retriever.embed_queries(texts="How are you?")
)
response = new_document_store.query_by_embedding(
    query_embedding, 
    top_k=1, 
    return_embedding=False
)

print(response[0].meta['answer'])
print(response[0].meta['id_video'])
```

    06/13/2021 09:57:59 - WARNING - farm.data_handler.processor -   Currently no support in InferenceProcessor for returning problematic ids
    Inferencing Samples: 100%|██████████| 1/1 [00:01<00:00,  1.25s/ Batches]

    Pretty good, thank you!
    77157368d489465a3af172497e80ed59


    


### Dialogue Mgr can stop here

Below is evaluation and further research


```python
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

sql_url = "postgresql:///ironman:kolomino@localhost:5432"

avatar_id = "avatar_id1334"

avatar_sql_url = sql_url.format(avatar_id)

engine = create_engine(sql_url)

if not database_exists("postgresql:///{}".format(avatar_id)):
    create_database("postgresql:///{}".format(avatar_id))

database_exists('postgresql:///margarita1234')  
```




    True




```python
%%capture --no-stdout --no-display

df_dial = pd.read_csv("data/DIALOGUES.csv", encoding='utf-8')
df_dial = df_dial[df_dial['Experiment'] == 'TEST']
df_dial
df_dial_test = df_dial.sample(frac=.5, random_state=1)
df_dial_test.reset_index()
df_dial_finetune = df_dial.drop(df_dial_test.index)

annotation_cols = ['BA1', 'BA2', 'BA3', 'BA4', 'BA5', 'BA6']

test_questions = df_dial_test['Q'].to_list()
test_questions_emb = retriever.embed_queries(texts=test_questions)

finetune_questions = df_dial_finetune['Q'].to_list()
finetune_questions_emb = retriever.embed_queries(texts=finetune_questions)

def hitsatk(k, document_store, test_questions, test_questions_emb, df_dial, annotation_cols, finder=None):
    
    hits_at_k = 0
    hits_at_k_itemized, probs, scores, answers = [], [], [], []
    
    for question, embedding in zip(test_questions, test_questions_emb):
        if finder == None:
            predictions = document_store.query_by_embedding(
                np.array(embedding), 
                top_k=k, 
                return_embedding=False
            )
            annotated_answers = df_dial[df_dial['Q'] == question][annotation_cols].values
            pred_answers = [pred.meta['answer'] for pred in predictions]
            probs.append(predictions[0].probability)
            scores.append(predictions[0].score)
            answers.append(pred_answers[0])
            
        else:
            predictions = finder.get_answers_via_similar_questions(
                question=question,
                top_k_retriever=k
            )
            annotated_answers = df_dial[df_dial['Q'] == embedding][annotation_cols].values
            if len(predictions["answers"]) == 0:
                pred_answers = ["NA"]
                probs.append(np.nan)
                scores.append(np.nan)
                answers.append("NA")
            else:
                pred_answers = [pred["answer"] for pred in predictions["answers"]]
                probs.append(predictions["answers"][0]["probability"])
                scores.append(predictions["answers"][0]["score"])
                answers.append(pred_answers[0])
            
        if any([pred_ans in annotated_answers for pred_ans in pred_answers]):
            hits_at_k += 1
            hits_at_k_itemized.append(1)
        else:
            hits_at_k += 0
            hits_at_k_itemized.append(0)
            
    return hits_at_k, hits_at_k_itemized, probs, scores, answers
```

    02/04/2021 12:18:34 - WARNING - farm.data_handler.processor -   Currently no support in InferenceProcessor for returning problematic ids
    02/04/2021 12:21:48 - WARNING - farm.data_handler.processor -   Currently no support in InferenceProcessor for returning problematic ids



```python
hits_at_k, hits_at_k_itemized, probs, scores, answers = hitsatk(
    1, new_document_store, test_questions, test_questions, df_dial_test_emb, annotation_cols)
```


```python
print("SR@1: ", hits_at_k/len(test_questions))

# All dialogues, hits @ 1: 0.15673981191222572
# Only PER dialogues, hits @ 1: 0.1569767441860465
# Only dialogues 6, 7 (2x PER, 2x UNI), hits @ 1: 0.19166666666666668
```


```python
for k in [2, 5, 10, 20]:
    hits_at_k, _, _, _, _ = hitsatk(
        k, new_document_store, test_questions, test_questions_emb, df_dial_test, annotation_cols)

    print("SR@{}: ".format(k), hits_at_k/len(test_questions))

# All dialogues, hits @ 10: 0.32601880877742945
# Only PER dialogues, hits @ 10: 0.3313953488372093
# Only dialogues 6, 7 (2x PER, 2x UNI), hits @ 10: 0.44166666666666665
```


```python
for k in [1, 2, 5, 10, 20]:
    hits_at_k, _, _, _, _ = hitsatk(
        k, new_document_store, finetune_questions, finetune_questions, df_dial_finetune, annotation_cols)

    print("SR@{}: ".format(k), hits_at_k/len(df_dial_finetune))
```


```python
df_thresholds = pd.DataFrame(
{
    "question": test_questions,
    "answer": answers,
    "hit_at_1": hits_at_k_itemized,
    "prob": probs,
    "score": scores,
    "no_ans": df_dial_test.BA1.isna()
})

df_thresholds["combo_mult"] = df_thresholds["prob"] * df_thresholds["score"]

df_thresholds["combo_sum"] = df_thresholds["prob"] + df_thresholds["score"]
```


```python
df_thresholds[["combo_mult", "combo_sum", "hit_at_1"]].groupby("hit_at_1").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">combo_mult</th>
      <th colspan="8" halign="left">combo_sum</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>hit_at_1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>136.0</td>
      <td>129.479534</td>
      <td>41.593143</td>
      <td>50.888723</td>
      <td>102.185536</td>
      <td>127.499947</td>
      <td>153.279563</td>
      <td>268.043406</td>
      <td>136.0</td>
      <td>156.343572</td>
      <td>39.948947</td>
      <td>75.620083</td>
      <td>130.812713</td>
      <td>155.479375</td>
      <td>179.754178</td>
      <td>284.688806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24.0</td>
      <td>158.622288</td>
      <td>53.439649</td>
      <td>82.381546</td>
      <td>120.849069</td>
      <td>157.822374</td>
      <td>186.660895</td>
      <td>263.675460</td>
      <td>24.0</td>
      <td>183.838438</td>
      <td>49.963871</td>
      <td>110.596163</td>
      <td>149.069217</td>
      <td>183.963621</td>
      <td>210.496181</td>
      <td>280.692296</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_thresholds.loc[df_thresholds["no_ans"]==True, "hit_at_1"] = 2

df_thresholds[["combo_mult", "combo_sum", "hit_at_1"]].groupby("hit_at_1").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">combo_mult</th>
      <th colspan="8" halign="left">combo_sum</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>hit_at_1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>115.0</td>
      <td>130.004457</td>
      <td>42.434080</td>
      <td>50.888723</td>
      <td>106.132912</td>
      <td>128.769637</td>
      <td>153.462122</td>
      <td>268.043406</td>
      <td>115.0</td>
      <td>156.799805</td>
      <td>40.801738</td>
      <td>75.620083</td>
      <td>134.731564</td>
      <td>156.691425</td>
      <td>179.923980</td>
      <td>284.688806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24.0</td>
      <td>158.622288</td>
      <td>53.439649</td>
      <td>82.381546</td>
      <td>120.849069</td>
      <td>157.822374</td>
      <td>186.660895</td>
      <td>263.675460</td>
      <td>24.0</td>
      <td>183.838438</td>
      <td>49.963871</td>
      <td>110.596163</td>
      <td>149.069217</td>
      <td>183.963621</td>
      <td>210.496181</td>
      <td>280.692296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.0</td>
      <td>126.604953</td>
      <td>37.462739</td>
      <td>77.118910</td>
      <td>92.068428</td>
      <td>120.260877</td>
      <td>148.255610</td>
      <td>224.495285</td>
      <td>21.0</td>
      <td>153.845151</td>
      <td>35.713321</td>
      <td>105.035724</td>
      <td>120.610835</td>
      <td>148.529077</td>
      <td>175.071212</td>
      <td>244.972127</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python

```

hits@1 = 0.3911764705882353

hits@2 = 0.4588235294117647

hits@5 = 0.4970588235294118

hits@10 = 0.5617647058823529

hits@20 = 0.611764705882353

hits@100 = 0.7794117647058824

hits@200 = 0.8558823529411764


```python
# for i in range(len(test_questions)):
#     if items_hits_at_k[i] == 0:
#         print(test_questions[i])
```

#### Add q-A Relevance


```python
preds = pd.read_csv('/Users/amc/Documents/glue_data/Margarita_1_100_ratio/test_results_mrpc.txt', 
                    sep='\t', encoding='utf-8')['prediction'].values
valid_df2valid_preds = pd.read_csv('~/Documents/TOIA-NYUAD/research/data/test_dev2test_preds.tsv', sep='\t', encoding='utf-8')

valid_preds = pd.DataFrame(
    {'q': valid_df2valid_preds['#1 String'].values, 
     'A': valid_df2valid_preds['#2 String'].values, 
     'y_pred': preds})


```


```python
def reranked_hitsatk(k, j, document_store, test_questions, test_questions_emb, df_dial, annotation_cols):
    result_items = []
    result_probs = []
    for question, embedding in zip(test_questions, test_questions_emb):
        predictions = document_store.query_by_embedding(
            np.array(embedding), 
            top_k=j, 
            return_embedding=False
        )
        pred_answers = [pred.meta['answer'] for pred in predictions]
        qq_probs = np.array([pred.probability for pred in predictions])
        qa_probs = np.array([valid_preds[(valid_preds['q']==question) &
                                (valid_preds['A']==pred_ans)]['y_pred'].values[0] for 
                    pred_ans in pred_answers])
        comb_probs = qq_probs * qa_probs
        sorted_probs = np.sort((comb_probs))[::-1][:k]
        sorted_indices = np.argsort((comb_probs))[::-1][:k]
        pred_answers_reranked = [pred_answers[i] for i in sorted_indices]
        if any([pred_ans in df_dial[
            df_dial['Q'] == question][annotation_cols].values 
                for pred_ans in pred_answers_reranked]):
            result_items.append(1)
        else:
            result_items.append(0)
        result_probs.append(sorted_probs[0])
    return result_items, result_probs
```


```python
for k in [1, 2, 5, 10, 20]:
    rr_hits_at_k, rr_probs = reranked_hitsatk(
        k, 10, new_document_store, test_questions, test_questions_emb, df_dial_test, annotation_cols)

    print("SR@{}_10: ".format(k), sum(rr_hits_at_k)/len(test_questions))
```

    SR@1_10:  0.20625
    SR@2_10:  0.25
    SR@5_10:  0.3125
    SR@10_10:  0.3375
    SR@20_10:  0.3375



```python
for k in [1, 2, 5, 10, 20]:
    rr_hits_at_k, rr_probs = reranked_hitsatk(
        k, 349, new_document_store, test_questions, test_questions_emb, df_dial_test, annotation_cols)

    print("SR@{}_300: ".format(k), sum(rr_hits_at_k)/len(test_questions))
```

    SR@1_300:  0.21875
    SR@2_300:  0.28125
    SR@5_300:  0.39375
    SR@10_300:  0.46875
    SR@20_300:  0.5375


hits@1_10 = 0.45294117647058824 | multiplying 0.4588235294117647 | weighted sum (.5 * qq) 0.45588235294117646

- test set = 0.24166666666666667 | 0.2833333333333333

hits@2_10 = 0.5088235294117647 | multiplying 0.49411764705882355

- test mult: 0.3333333333333333

hits@5_10 = 0.5323529411764706 | multiplying 0.5323529411764706

- test mult: 0.4 hit@10_10 -- 0.44166666666666665

hits@10_200 = 0.6558823529411765

This is basically scoring all and summing scores:

hits@1_300 = 0.5411764705882353 | multiplying 0.5352941176470588

- test set = 0.21666666666666667 | 0.24166666666666667

hits@10_300 = 0.6794117647058824 | multiplying 0.7176470588235294


#### Worth trying to filter with Q-A and re-ranking with Q-Q.


```python

rr_hits_at_k, rr_probs = reranked_hitsatk(
    1, 10, new_document_store, test_questions, test_questions_emb, df_dial_test, annotation_cols)

```


```python
df_rr_thr = pd.DataFrame(
{
    "question": test_questions,
    "answer": answers,
    "hit_at_k": rr_hits_at_k,
    "prob": rr_probs,
    "no_ans": df_dial_test.BA1.isna()
})
```


```python
df_rr_thr.groupby("hit_at_k").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">prob</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>hit_at_k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>127.0</td>
      <td>0.225602</td>
      <td>0.348523</td>
      <td>0.000000</td>
      <td>0.000008</td>
      <td>0.000168</td>
      <td>0.706131</td>
      <td>0.915177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33.0</td>
      <td>0.598176</td>
      <td>0.382590</td>
      <td>0.000007</td>
      <td>0.004419</td>
      <td>0.810861</td>
      <td>0.849148</td>
      <td>0.942493</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_rr_thr.loc[df_thresholds["no_ans"]==True, "hit_at_k"] = 2

df_rr_thr.groupby("hit_at_k").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">prob</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>hit_at_k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>106.0</td>
      <td>0.231319</td>
      <td>0.353010</td>
      <td>0.000000</td>
      <td>0.000008</td>
      <td>0.000147</td>
      <td>0.708181</td>
      <td>0.915177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33.0</td>
      <td>0.598176</td>
      <td>0.382590</td>
      <td>0.000007</td>
      <td>0.004419</td>
      <td>0.810861</td>
      <td>0.849148</td>
      <td>0.942493</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.0</td>
      <td>0.196747</td>
      <td>0.331615</td>
      <td>0.000000</td>
      <td>0.000022</td>
      <td>0.000619</td>
      <td>0.245588</td>
      <td>0.834083</td>
    </tr>
  </tbody>
</table>
</div>




```python
thr_sel = df_rr_thr.groupby("hit_at_k").describe()["prob"]["25%"][1]
thr_sel
```




    0.004418585643219317




```python
def isNaN(string):
    return string != string

def reranked_hitsatk_thr(k, j, document_store, test_questions, test_questions_emb, df_dial, thr):
    result_items = []
    result_probs = []
    for question, embedding in zip(test_questions, test_questions_emb):
        predictions = document_store.query_by_embedding(
            np.array(embedding), 
            top_k=j, 
            return_embedding=False
        )
        pred_answers = [pred.meta['answer'] for pred in predictions]
        qq_probs = np.array([pred.probability for pred in predictions])
        qa_probs = np.array([valid_preds[(valid_preds['q']==question) &
                                (valid_preds['A']==pred_ans)]['y_pred'].values[0] for 
                    pred_ans in pred_answers])
        comb_probs = qq_probs * qa_probs        
        sorted_probs = np.sort((comb_probs))[::-1][:k]
        sorted_indices = np.argsort((comb_probs))[::-1][:k]
        pred_answers_reranked = [pred_answers[i] for i, p in zip(sorted_indices, sorted_probs) if p >= thr]
        annotated_answers = df_dial[df_dial['Q'] == question][annotation_cols].values
        if any([pred_ans in annotated_answers for pred_ans in pred_answers_reranked]):
            result_items.append(1)
            result_probs.append(sorted_probs[0])
        elif (len(pred_answers_reranked) == 0) & (isNaN(annotated_answers[0][0])):
            result_items.append(1)
            result_probs.append(1)
        else:
            result_items.append(0)
            result_probs.append(max(comb_probs))
    return result_items, result_probs
```


```python
rr_test_hits_at_k, rr_test_probs = reranked_hitsatk_thr(1, 10,
                                                        new_document_store, 
                                                        test_questions, 
                                                        test_questions_emb, 
                                                        df_dial_test, thr_sel)

print(sum(rr_test_hits_at_k)/len(test_questions))

```

    0.23125



```python
rr_finetune_hits_at_k, rr_finetune_probs = reranked_hitsatk_thr(1, 10, new_document_store, finetune_questions, finetune_questions_emb, df_dial_finetune, thr_sel)

print(sum(rr_finetune_hits_at_k)/len(finetune_questions))
```

    0.24528301886792453



```python
rr_finetune_hits_at_k, rr_finetune_probs = reranked_hitsatk_thr(1, 349, new_document_store, finetune_questions, finetune_questions_emb, df_dial_finetune, thr_sel)

print(sum(rr_finetune_hits_at_k)/len(finetune_questions))
```

    0.2138364779874214



```python

```


```python

```


```python

```

# Get dialogues in QA format For Fine Tuning


```python
import json

squadlike_dict = {"version": "v1.1",
                  "data": [{"title": "Margarita_squadFormat", "paragraphs": []}]}

print(json.dumps(squadlike_dict, indent = 2)) 
```

    {
      "version": "v1.1",
      "data": [
        {
          "title": "Margarita_squadFormat",
          "paragraphs": []
        }
      ]
    }



```python
i = 0

for text, answer in zip(df["text"], df["answer"]):
    squadlike_dict["data"][0]["paragraphs"].append(
        {"qas": [{
            "question": text,
            "id": 'id' + str(i),
            "answers": [{"text": answer, 
                         "answer_start": len(text) + 1}],
            "is_impossible": False}],
         "context": "{} {}".format(text, answer)})
    i += 1
```


```python
print(json.dumps(squadlike_dict, indent = 2)) 
```

    {
      "version": "v1.1",
      "data": [
        {
          "title": "Margarita_squadFormat",
          "paragraphs": [
            {
              "qas": [
                {
                  "question": "Please dance!",
                  "id": "id0",
                  "answers": [
                    {
                      "text": "Okay! (Dances)",
                      "answer_start": 14
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Please dance! Okay! (Dances)"
            },
            {
              "qas": [
                {
                  "question": "Please play something!",
                  "id": "id1",
                  "answers": [
                    {
                      "text": "Okay! (Plays ukulele)",
                      "answer_start": 23
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Please play something! Okay! (Plays ukulele)"
            },
            {
              "qas": [
                {
                  "question": "Please sing!",
                  "id": "id2",
                  "answers": [
                    {
                      "text": "Okay! (Sings)",
                      "answer_start": 13
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Please sing! Okay! (Sings)"
            },
            {
              "qas": [
                {
                  "question": "OK",
                  "id": "id3",
                  "answers": [
                    {
                      "text": "Tell me what else you'd like to know.",
                      "answer_start": 3
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "OK Tell me what else you'd like to know."
            },
            {
              "qas": [
                {
                  "question": "uh huh",
                  "id": "id4",
                  "answers": [
                    {
                      "text": "Would you like to know some more?",
                      "answer_start": 7
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "uh huh Would you like to know some more?"
            },
            {
              "qas": [
                {
                  "question": "Do you accept applicants who already hold a bachelor of arts or science degrees?",
                  "id": "id5",
                  "answers": [
                    {
                      "text": "At NYUAD there are only programs to get the first bachelor.",
                      "answer_start": 81
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you accept applicants who already hold a bachelor of arts or science degrees? At NYUAD there are only programs to get the first bachelor."
            },
            {
              "qas": [
                {
                  "question": "When will I hear if I get admitted?",
                  "id": "id6",
                  "answers": [
                    {
                      "text": "Check on the university's web page!",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "When will I hear if I get admitted? Check on the university's web page!"
            },
            {
              "qas": [
                {
                  "question": "How did you prepare to apply there?",
                  "id": "id7",
                  "answers": [
                    {
                      "text": "I googled a lot of examples of university applications. I made a list of all my achievements and everything I want to mention. And I just asked a lot of people to proofread and give me advice.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How did you prepare to apply there? I googled a lot of examples of university applications. I made a list of all my achievements and everything I want to mention. And I just asked a lot of people to proofread and give me advice."
            },
            {
              "qas": [
                {
                  "question": "And you got in, when?",
                  "id": "id8",
                  "answers": [
                    {
                      "text": "I was accepted to NYUAD in December.",
                      "answer_start": 22
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "And you got in, when? I was accepted to NYUAD in December."
            },
            {
              "qas": [
                {
                  "question": "When should I apply for NYU Abu Dhabi?",
                  "id": "id9",
                  "answers": [
                    {
                      "text": "I'd recommend you apply in October, a year before the university's academic year.",
                      "answer_start": 39
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "When should I apply for NYU Abu Dhabi? I'd recommend you apply in October, a year before the university's academic year."
            },
            {
              "qas": [
                {
                  "question": "Was it hard to apply?",
                  "id": "id10",
                  "answers": [
                    {
                      "text": "It was a very long application and then you had to come here for a candidate weekend and interview and then you would find out if you got in.",
                      "answer_start": 22
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Was it hard to apply? It was a very long application and then you had to come here for a candidate weekend and interview and then you would find out if you got in."
            },
            {
              "qas": [
                {
                  "question": "Am I able to change campus location after I enroll?",
                  "id": "id11",
                  "answers": [
                    {
                      "text": "It's difficult, but you could. Also, keep in mind that scholarships are different for each campus.",
                      "answer_start": 52
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Am I able to change campus location after I enroll? It's difficult, but you could. Also, keep in mind that scholarships are different for each campus."
            },
            {
              "qas": [
                {
                  "question": "Does NYUAD care about diversity?",
                  "id": "id12",
                  "answers": [
                    {
                      "text": "NYUAD cares about all types of diversity of its students, including of their backgrounds and diversity of thought.",
                      "answer_start": 33
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Does NYUAD care about diversity? NYUAD cares about all types of diversity of its students, including of their backgrounds and diversity of thought."
            },
            {
              "qas": [
                {
                  "question": "Is the TOEFL required for consideration for admission to NYU Abu Dhabi?",
                  "id": "id13",
                  "answers": [
                    {
                      "text": "NYUAD does not currently require an English language proficiency test. Your English skills will be obvious in the application and later on in the interview.",
                      "answer_start": 72
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is the TOEFL required for consideration for admission to NYU Abu Dhabi? NYUAD does not currently require an English language proficiency test. Your English skills will be obvious in the application and later on in the interview."
            },
            {
              "qas": [
                {
                  "question": "Did you apply to other universities, too?",
                  "id": "id14",
                  "answers": [
                    {
                      "text": "NYUAD was my first choice and I didn't even apply to other universities! But that's because I applied very early so I had plenty of time to figure out other universities in case I wasn't admitted.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you apply to other universities, too? NYUAD was my first choice and I didn't even apply to other universities! But that's because I applied very early so I had plenty of time to figure out other universities in case I wasn't admitted."
            },
            {
              "qas": [
                {
                  "question": "If I'm not admitted can I reapply?",
                  "id": "id15",
                  "answers": [
                    {
                      "text": "Of course you can! But I think only in the following year.",
                      "answer_start": 35
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "If I'm not admitted can I reapply? Of course you can! But I think only in the following year."
            },
            {
              "qas": [
                {
                  "question": "Can I apply to more than one NYU campus at a time?",
                  "id": "id16",
                  "answers": [
                    {
                      "text": "Sure, you can apply to all three!",
                      "answer_start": 51
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Can I apply to more than one NYU campus at a time? Sure, you can apply to all three!"
            },
            {
              "qas": [
                {
                  "question": "How does NYU Abu Dhabi decide which student to admit?",
                  "id": "id17",
                  "answers": [
                    {
                      "text": "The whole application is very important, they look at it holistically: your grades, extracurriculars, essays, and not one thing is more important than the other.",
                      "answer_start": 54
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How does NYU Abu Dhabi decide which student to admit? The whole application is very important, they look at it holistically: your grades, extracurriculars, essays, and not one thing is more important than the other."
            },
            {
              "qas": [
                {
                  "question": "I'm thinking about applying to university. It's not... It's not that university, but they put a lot of attention to the application letter. Did you have it there?",
                  "id": "id18",
                  "answers": [
                    {
                      "text": "We have an extensive application that includes details about the students and essays.",
                      "answer_start": 163
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "I'm thinking about applying to university. It's not... It's not that university, but they put a lot of attention to the application letter. Did you have it there? We have an extensive application that includes details about the students and essays."
            },
            {
              "qas": [
                {
                  "question": "Does NYU Abu Dhabi accept the common application?",
                  "id": "id19",
                  "answers": [
                    {
                      "text": "You apply to NYUAD through this website that's called Common Application which you can use to apply to multiple universities at the same time.",
                      "answer_start": 50
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Does NYU Abu Dhabi accept the common application? You apply to NYUAD through this website that's called Common Application which you can use to apply to multiple universities at the same time."
            },
            {
              "qas": [
                {
                  "question": "I am an undergraduate at another university, can I apply to NYU Abu Dhabi?",
                  "id": "id20",
                  "answers": [
                    {
                      "text": "You can definitely apply, but you would have to start as a freshman.",
                      "answer_start": 75
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "I am an undergraduate at another university, can I apply to NYU Abu Dhabi? You can definitely apply, but you would have to start as a freshman."
            },
            {
              "qas": [
                {
                  "question": "Which standardized test scores do I need to submit for my application?",
                  "id": "id21",
                  "answers": [
                    {
                      "text": "You can send official results of any nationally or internationally recognized standardized tests.",
                      "answer_start": 71
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Which standardized test scores do I need to submit for my application? You can send official results of any nationally or internationally recognized standardized tests."
            },
            {
              "qas": [
                {
                  "question": "Can I apply online?",
                  "id": "id22",
                  "answers": [
                    {
                      "text": "You apply to NYUAD through this website that's called Common Application which you can use for other universities at the same time.",
                      "answer_start": 20
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Can I apply online? You apply to NYUAD through this website that's called Common Application which you can use for other universities at the same time."
            },
            {
              "qas": [
                {
                  "question": "So you applied in October, right?",
                  "id": "id23",
                  "answers": [
                    {
                      "text": "Yes.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "So you applied in October, right? Yes."
            },
            {
              "qas": [
                {
                  "question": "Did you have a connection to someone from the university so you could give this the essay to them and just ask them to proofread?",
                  "id": "id24",
                  "answers": [
                    {
                      "text": "I always wanted to go abroad for my education. My brother happened to know another Moldovan friend who was studying here and told me about the full ride scholarship, that Abu Dhabi is safe, that there are lots of traveling opportunities and the people are great, so it was a no-brainer.",
                      "answer_start": 130
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you have a connection to someone from the university so you could give this the essay to them and just ask them to proofread? I always wanted to go abroad for my education. My brother happened to know another Moldovan friend who was studying here and told me about the full ride scholarship, that Abu Dhabi is safe, that there are lots of traveling opportunities and the people are great, so it was a no-brainer."
            },
            {
              "qas": [
                {
                  "question": "What are you?",
                  "id": "id25",
                  "answers": [
                    {
                      "text": "I'm an avatar of Margarita!",
                      "answer_start": 14
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are you? I'm an avatar of Margarita!"
            },
            {
              "qas": [
                {
                  "question": "Do you prefer the Abu Dhabi campus over the other sites?",
                  "id": "id26",
                  "answers": [
                    {
                      "text": "I love all campuses for different reasons. The other two are in the middle of the city and are more lively, but our campus has the best facilities and is less crowded which makes it easier to focus on studying.",
                      "answer_start": 57
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you prefer the Abu Dhabi campus over the other sites? I love all campuses for different reasons. The other two are in the middle of the city and are more lively, but our campus has the best facilities and is less crowded which makes it easier to focus on studying."
            },
            {
              "qas": [
                {
                  "question": "Compare the academic stress levels between the two campuses.",
                  "id": "id27",
                  "answers": [
                    {
                      "text": "I think the Abu Dhabi and the Shanghai campuses are more intense in terms of studying compared to the New York campus.",
                      "answer_start": 61
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Compare the academic stress levels between the two campuses. I think the Abu Dhabi and the Shanghai campuses are more intense in terms of studying compared to the New York campus."
            },
            {
              "qas": [
                {
                  "question": "What is a J-Term?",
                  "id": "id28",
                  "answers": [
                    {
                      "text": "J-Term or January Term is a month long intensive course and can be done at one of the global sites.",
                      "answer_start": 18
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is a J-Term? J-Term or January Term is a month long intensive course and can be done at one of the global sites."
            },
            {
              "qas": [
                {
                  "question": "Where is the NYUAD campus located?",
                  "id": "id29",
                  "answers": [
                    {
                      "text": "Our campus is located on an island and it's very different from any other university. It's a very tight knit community as we only have over one thousand students of all majors, and since the campus is a bit far from the city, students spend a lot of time together and most of them live on campus.",
                      "answer_start": 35
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where is the NYUAD campus located? Our campus is located on an island and it's very different from any other university. It's a very tight knit community as we only have over one thousand students of all majors, and since the campus is a bit far from the city, students spend a lot of time together and most of them live on campus."
            },
            {
              "qas": [
                {
                  "question": "What are the roles of resident assistants?",
                  "id": "id30",
                  "answers": [
                    {
                      "text": "RAs are the backbone of a strong community: they organize events, act as counselors for students and enforce dorm policies.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are the roles of resident assistants? RAs are the backbone of a strong community: they organize events, act as counselors for students and enforce dorm policies."
            },
            {
              "qas": [
                {
                  "question": "How far are you guys from the city?",
                  "id": "id31",
                  "answers": [
                    {
                      "text": "We are 30 minutes away from the city center by car or bus.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How far are you guys from the city? We are 30 minutes away from the city center by car or bus."
            },
            {
              "qas": [
                {
                  "question": "Can I choose to live with my friend in the dorms?",
                  "id": "id32",
                  "answers": [
                    {
                      "text": "Yes, you can choose your roommates!",
                      "answer_start": 50
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Can I choose to live with my friend in the dorms? Yes, you can choose your roommates!"
            },
            {
              "qas": [
                {
                  "question": "Where is the campus?",
                  "id": "id33",
                  "answers": [
                    {
                      "text": "Our campus is located on an island and it's very different from any other university. It's a tight knit community with only over one thousand students of all majors, and since it's a bit far from the city, students spend a lot of time together and most of them live on campus.",
                      "answer_start": 21
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where is the campus? Our campus is located on an island and it's very different from any other university. It's a tight knit community with only over one thousand students of all majors, and since it's a bit far from the city, students spend a lot of time together and most of them live on campus."
            },
            {
              "qas": [
                {
                  "question": "OK, first off how would you describe the Abu Dhabi campus?",
                  "id": "id34",
                  "answers": [
                    {
                      "text": "Studying at NYUAD is lovely! Good-quality people, a lot of things to do on campus, great weather and lots of traveling.",
                      "answer_start": 59
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "OK, first off how would you describe the Abu Dhabi campus? Studying at NYUAD is lovely! Good-quality people, a lot of things to do on campus, great weather and lots of traveling."
            },
            {
              "qas": [
                {
                  "question": "I like your *",
                  "id": "id35",
                  "answers": [
                    {
                      "text": "Awww thanks!",
                      "answer_start": 14
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "I like your * Awww thanks!"
            },
            {
              "qas": [
                {
                  "question": "What is the best compliment you can receive?",
                  "id": "id36",
                  "answers": [
                    {
                      "text": "Somebody told me recently that I am always \"full of life\" and I'm still high off that compliment.",
                      "answer_start": 45
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the best compliment you can receive? Somebody told me recently that I am always \"full of life\" and I'm still high off that compliment."
            },
            {
              "qas": [
                {
                  "question": "Nice!",
                  "id": "id37",
                  "answers": [
                    {
                      "text": "Thank you so much!",
                      "answer_start": 6
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Nice! Thank you so much!"
            },
            {
              "qas": [
                {
                  "question": "What is the best compliment you have ever received?",
                  "id": "id38",
                  "answers": [
                    {
                      "text": "Somebody told me lately that I am always \"full of life\" and I'm still high off that compliment.",
                      "answer_start": 52
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the best compliment you have ever received? Somebody told me lately that I am always \"full of life\" and I'm still high off that compliment."
            },
            {
              "qas": [
                {
                  "question": "Oh that's impressive.",
                  "id": "id39",
                  "answers": [
                    {
                      "text": "Thank you!",
                      "answer_start": 22
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Oh that's impressive. Thank you!"
            },
            {
              "qas": [
                {
                  "question": "How are you enjoying Boxing?",
                  "id": "id40",
                  "answers": [
                    {
                      "text": "I completely love this class!",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How are you enjoying Boxing? I completely love this class!"
            },
            {
              "qas": [
                {
                  "question": "What was it about one of those courses, negotiations say, that you liked?",
                  "id": "id41",
                  "answers": [
                    {
                      "text": "Negotiation and Consensus Building teaches how to protect your interests when you negotiate with somebody while still building relationships. I loved it because I implement it in my daily life!",
                      "answer_start": 74
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was it about one of those courses, negotiations say, that you liked? Negotiation and Consensus Building teaches how to protect your interests when you negotiate with somebody while still building relationships. I loved it because I implement it in my daily life!"
            },
            {
              "qas": [
                {
                  "question": "What are the most fun classes you took at NYUAD?",
                  "id": "id42",
                  "answers": [
                    {
                      "text": "Some of my favorite ones were Conducting, Composing for Film & Multimedia, & Advanced Song-Writing.",
                      "answer_start": 49
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are the most fun classes you took at NYUAD? Some of my favorite ones were Conducting, Composing for Film & Multimedia, & Advanced Song-Writing."
            },
            {
              "qas": [
                {
                  "question": "What was your favorite class?",
                  "id": "id43",
                  "answers": [
                    {
                      "text": "My favorite course was Negotiation and Consensus Building in New York and Fundamentals of Acting in Abu Dhabi.",
                      "answer_start": 30
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was your favorite class? My favorite course was Negotiation and Consensus Building in New York and Fundamentals of Acting in Abu Dhabi."
            },
            {
              "qas": [
                {
                  "question": "How did you adapt to the Arabic culture?",
                  "id": "id44",
                  "answers": [
                    {
                      "text": "It was quite a smooth transition because I'm open-minded and people here are friendly!",
                      "answer_start": 41
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How did you adapt to the Arabic culture? It was quite a smooth transition because I'm open-minded and people here are friendly!"
            },
            {
              "qas": [
                {
                  "question": "And did you like it?",
                  "id": "id45",
                  "answers": [
                    {
                      "text": "Can we change the subject? *smiles shyly*",
                      "answer_start": 21
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "And did you like it? Can we change the subject? *smiles shyly*"
            },
            {
              "qas": [
                {
                  "question": "What are the negative aspects of NYUAD?",
                  "id": "id46",
                  "answers": [
                    {
                      "text": "Relationships are harder to maintain because of all the study-aways. Also, the campus is a bit isolated from the city so it's not as lively.",
                      "answer_start": 40
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are the negative aspects of NYUAD? Relationships are harder to maintain because of all the study-aways. Also, the campus is a bit isolated from the city so it's not as lively."
            },
            {
              "qas": [
                {
                  "question": "Which parent are you closer to and why?",
                  "id": "id47",
                  "answers": [
                    {
                      "text": "I am closer to my mom because she understand me better, but I love my dad just as much.",
                      "answer_start": 40
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Which parent are you closer to and why? I am closer to my mom because she understand me better, but I love my dad just as much."
            },
            {
              "qas": [
                {
                  "question": "Do you have any siblings?",
                  "id": "id48",
                  "answers": [
                    {
                      "text": "I have a brother.",
                      "answer_start": 26
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have any siblings? I have a brother."
            },
            {
              "qas": [
                {
                  "question": "What does your brother do?",
                  "id": "id49",
                  "answers": [
                    {
                      "text": "My brother is a programmer.",
                      "answer_start": 27
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What does your brother do? My brother is a programmer."
            },
            {
              "qas": [
                {
                  "question": "What do your parents do?",
                  "id": "id50",
                  "answers": [
                    {
                      "text": "My mom is a doctor, and my dad is a notary.",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What do your parents do? My mom is a doctor, and my dad is a notary."
            },
            {
              "qas": [
                {
                  "question": "Were your parents okay with you studying in the Middle East?",
                  "id": "id51",
                  "answers": [
                    {
                      "text": "My parents weren't okay with me studying here at first, but I assured them that it's safe and they were able to see all the benefits as well.",
                      "answer_start": 61
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Were your parents okay with you studying in the Middle East? My parents weren't okay with me studying here at first, but I assured them that it's safe and they were able to see all the benefits as well."
            },
            {
              "qas": [
                {
                  "question": "Do you have any musicians in your family?",
                  "id": "id52",
                  "answers": [
                    {
                      "text": "Not really, but my mom and grandma used to like to sing.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have any musicians in your family? Not really, but my mom and grandma used to like to sing."
            },
            {
              "qas": [
                {
                  "question": "Yeah. Do you have like... little nephews and nieces..?",
                  "id": "id53",
                  "answers": [
                    {
                      "text": "Not yet! Maybe one day!",
                      "answer_start": 55
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Yeah. Do you have like... little nephews and nieces..? Not yet! Maybe one day!"
            },
            {
              "qas": [
                {
                  "question": "Is your brother older or younger than you?",
                  "id": "id54",
                  "answers": [
                    {
                      "text": "So my family consists of four people. It's my mom, my dad and my brother. My brother is older and he is a software engineer in Moldova.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is your brother older or younger than you? So my family consists of four people. It's my mom, my dad and my brother. My brother is older and he is a software engineer in Moldova."
            },
            {
              "qas": [
                {
                  "question": "Nice. And Have your parents visit you here?",
                  "id": "id55",
                  "answers": [
                    {
                      "text": "Yes, they have! They visited me in my junior year and for graduation!",
                      "answer_start": 44
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Nice. And Have your parents visit you here? Yes, they have! They visited me in my junior year and for graduation!"
            },
            {
              "qas": [
                {
                  "question": "And are you a coffee person or a tea person?",
                  "id": "id56",
                  "answers": [
                    {
                      "text": "Both! I love tea because it's so healthy. I love coffee for its taste and smell.",
                      "answer_start": 45
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "And are you a coffee person or a tea person? Both! I love tea because it's so healthy. I love coffee for its taste and smell."
            },
            {
              "qas": [
                {
                  "question": "What is your favorite color?",
                  "id": "id57",
                  "answers": [
                    {
                      "text": "I like both turquoise and yellow.",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your favorite color? I like both turquoise and yellow."
            },
            {
              "qas": [
                {
                  "question": "What kind of sushi do you like?",
                  "id": "id58",
                  "answers": [
                    {
                      "text": "I like different kinds of sushi, but mostly the ones with salmon.",
                      "answer_start": 32
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What kind of sushi do you like? I like different kinds of sushi, but mostly the ones with salmon."
            },
            {
              "qas": [
                {
                  "question": "Where do you usually go with your friends?",
                  "id": "id59",
                  "answers": [
                    {
                      "text": "I like to go to the mall, eat out or hang at the beach with my friends.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where do you usually go with your friends? I like to go to the mall, eat out or hang at the beach with my friends."
            },
            {
              "qas": [
                {
                  "question": "What's your favorite type of foreign food?",
                  "id": "id60",
                  "answers": [
                    {
                      "text": "I love Chinese Hot Pot and sushi!",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your favorite type of foreign food? I love Chinese Hot Pot and sushi!"
            },
            {
              "qas": [
                {
                  "question": "Are there any other interests outside those two? Music and economics.",
                  "id": "id61",
                  "answers": [
                    {
                      "text": "I love drawing sometimes, but abstract shapes, basically doodling. I love volleyball and biking a lot and I am very interested in negotiation as a science and the aspects of communication.",
                      "answer_start": 70
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are there any other interests outside those two? Music and economics. I love drawing sometimes, but abstract shapes, basically doodling. I love volleyball and biking a lot and I am very interested in negotiation as a science and the aspects of communication."
            },
            {
              "qas": [
                {
                  "question": "What's your favorite food?",
                  "id": "id62",
                  "answers": [
                    {
                      "text": "I love eating solyanka and poke bowls!",
                      "answer_start": 27
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your favorite food? I love eating solyanka and poke bowls!"
            },
            {
              "qas": [
                {
                  "question": "What's your favorite dessert?",
                  "id": "id63",
                  "answers": [
                    {
                      "text": "I love fruit chocolate and... no, that's not a thing! Fruit salads and milk chocolate. Sorry.",
                      "answer_start": 30
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your favorite dessert? I love fruit chocolate and... no, that's not a thing! Fruit salads and milk chocolate. Sorry."
            },
            {
              "qas": [
                {
                  "question": "What's your favorite drink?",
                  "id": "id64",
                  "answers": [
                    {
                      "text": "I love orange juice! And wine. I love wine.",
                      "answer_start": 28
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your favorite drink? I love orange juice! And wine. I love wine."
            },
            {
              "qas": [
                {
                  "question": "What's your favorite genre of book?",
                  "id": "id65",
                  "answers": [
                    {
                      "text": "I love science-fiction, comedy and spooky things.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your favorite genre of book? I love science-fiction, comedy and spooky things."
            },
            {
              "qas": [
                {
                  "question": "What is your favorite author?",
                  "id": "id66",
                  "answers": [
                    {
                      "text": "I love Veronica Roth.",
                      "answer_start": 30
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your favorite author? I love Veronica Roth."
            },
            {
              "qas": [
                {
                  "question": "What sports do you like?",
                  "id": "id67",
                  "answers": [
                    {
                      "text": "I love volleyball, biking, boxing and rock-climbing.",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What sports do you like? I love volleyball, biking, boxing and rock-climbing."
            },
            {
              "qas": [
                {
                  "question": "Do you miss the food, the Moldovan cuisine?",
                  "id": "id68",
                  "answers": [
                    {
                      "text": "I used to more then, but now I got so used to just changing my diet depending on where I am at that time. And also I found some place where we can eat Russian food and the dining hall is also making Russian food from time to time... so I don't miss it that much.",
                      "answer_start": 44
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you miss the food, the Moldovan cuisine? I used to more then, but now I got so used to just changing my diet depending on where I am at that time. And also I found some place where we can eat Russian food and the dining hall is also making Russian food from time to time... so I don't miss it that much."
            },
            {
              "qas": [
                {
                  "question": "Which course did you like best?",
                  "id": "id69",
                  "answers": [
                    {
                      "text": "My favorite courses were Negotiation and Consensus Building in New York and Fundamentals of Acting in Abu Dhabi.",
                      "answer_start": 32
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Which course did you like best? My favorite courses were Negotiation and Consensus Building in New York and Fundamentals of Acting in Abu Dhabi."
            },
            {
              "qas": [
                {
                  "question": "What city would you most like to live in?",
                  "id": "id70",
                  "answers": [
                    {
                      "text": "My favorite place ever is New York. I hope to get back there at some point. I just love how vibrant the city is and there's always something to do and people are just the right amount of polite and minding their own business.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What city would you most like to live in? My favorite place ever is New York. I hope to get back there at some point. I just love how vibrant the city is and there's always something to do and people are just the right amount of polite and minding their own business."
            },
            {
              "qas": [
                {
                  "question": "What's your favorite sport?",
                  "id": "id71",
                  "answers": [
                    {
                      "text": "My favorite sport is definitely volleyball.",
                      "answer_start": 28
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your favorite sport? My favorite sport is definitely volleyball."
            },
            {
              "qas": [
                {
                  "question": "Do you like local food?",
                  "id": "id72",
                  "answers": [
                    {
                      "text": "Sometimes it is a little spicy for me. But in general if I tone it down a bit it's very delicious.",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you like local food? Sometimes it is a little spicy for me. But in general if I tone it down a bit it's very delicious."
            },
            {
              "qas": [
                {
                  "question": "Have you tried Arabic coffee?",
                  "id": "id73",
                  "answers": [
                    {
                      "text": "Yes. Arabic coffee is so strong for me so I have to dilute it. But it's very tasty and I love the tradition of - you know - small cups and stuff... I love it.",
                      "answer_start": 30
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you tried Arabic coffee? Yes. Arabic coffee is so strong for me so I have to dilute it. But it's very tasty and I love the tradition of - you know - small cups and stuff... I love it."
            },
            {
              "qas": [
                {
                  "question": "How's the meal plan?",
                  "id": "id74",
                  "answers": [
                    {
                      "text": "It's pretty good. We get university money that we can only spend on campus and you get some money that you can spend outside.",
                      "answer_start": 21
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How's the meal plan? It's pretty good. We get university money that we can only spend on campus and you get some money that you can spend outside."
            },
            {
              "qas": [
                {
                  "question": "Wait even if you're an international student, do you get a generous financial aid?",
                  "id": "id75",
                  "answers": [
                    {
                      "text": "Yes, exactly.",
                      "answer_start": 83
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Wait even if you're an international student, do you get a generous financial aid? Yes, exactly."
            },
            {
              "qas": [
                {
                  "question": "What is financial aid?",
                  "id": "id76",
                  "answers": [
                    {
                      "text": "Many students here are on full scholarship, but all of them have at least partial scholarships.",
                      "answer_start": 23
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is financial aid? Many students here are on full scholarship, but all of them have at least partial scholarships."
            },
            {
              "qas": [
                {
                  "question": "Could you please describe me your freshman year. Like, how was it?",
                  "id": "id77",
                  "answers": [
                    {
                      "text": "It was challenging to find a tight group of friends because everyone was kind of sticking to their own cultures, because in this university there are people from everywhere around the world. But I loved it because I learned a lot.",
                      "answer_start": 67
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Could you please describe me your freshman year. Like, how was it? It was challenging to find a tight group of friends because everyone was kind of sticking to their own cultures, because in this university there are people from everywhere around the world. But I loved it because I learned a lot."
            },
            {
              "qas": [
                {
                  "question": "How do you make money with music?",
                  "id": "id78",
                  "answers": [
                    {
                      "text": "By being good at what you do and knowing people. That's how you get a job in the music industry and you grow from there.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do you make money with music? By being good at what you do and knowing people. That's how you get a job in the music industry and you grow from there."
            },
            {
              "qas": [
                {
                  "question": "Because like now it's like your post grad period. I know we've talked about a bit but like do you actually see yourself staying in this country for a longer time?",
                  "id": "id79",
                  "answers": [
                    {
                      "text": "I love the UAE, I don't see myself staying forever in any place ever. I love living in different places.",
                      "answer_start": 163
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Because like now it's like your post grad period. I know we've talked about a bit but like do you actually see yourself staying in this country for a longer time? I love the UAE, I don't see myself staying forever in any place ever. I love living in different places."
            },
            {
              "qas": [
                {
                  "question": "And what do you want to do after university?",
                  "id": "id80",
                  "answers": [
                    {
                      "text": "I will be working in Tax, Transfer Pricing, at PwC starting August 2019!",
                      "answer_start": 45
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "And what do you want to do after university? I will be working in Tax, Transfer Pricing, at PwC starting August 2019!"
            },
            {
              "qas": [
                {
                  "question": "Do you plan on going to grad school afterwards?",
                  "id": "id81",
                  "answers": [
                    {
                      "text": "I would love to get an MBA!",
                      "answer_start": 48
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you plan on going to grad school afterwards? I would love to get an MBA!"
            },
            {
              "qas": [
                {
                  "question": "What do you think you'll be doing in 10 years?",
                  "id": "id82",
                  "answers": [
                    {
                      "text": "In 10 years I hope to have my own business. I definitely want to contribute to the world in a way, I am very passionate about climate change, overpopulation issues, philanthropy in general. So I hope to be a businesswoman and have an impact on the world.",
                      "answer_start": 47
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What do you think you'll be doing in 10 years? In 10 years I hope to have my own business. I definitely want to contribute to the world in a way, I am very passionate about climate change, overpopulation issues, philanthropy in general. So I hope to be a businesswoman and have an impact on the world."
            },
            {
              "qas": [
                {
                  "question": "So do you envision yourself going back to your country as part of your professional career?",
                  "id": "id83",
                  "answers": [
                    {
                      "text": "Maybe in the future. Not now for sure. But maybe in ten years. Yes.",
                      "answer_start": 92
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "So do you envision yourself going back to your country as part of your professional career? Maybe in the future. Not now for sure. But maybe in ten years. Yes."
            },
            {
              "qas": [
                {
                  "question": "Then what do you want to do in the future?",
                  "id": "id84",
                  "answers": [
                    {
                      "text": "My goal in life is to learn as much as possible, support my family back home and make the world a better place.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Then what do you want to do in the future? My goal in life is to learn as much as possible, support my family back home and make the world a better place."
            },
            {
              "qas": [
                {
                  "question": "Do you have a bucket list?",
                  "id": "id85",
                  "answers": [
                    {
                      "text": "No.",
                      "answer_start": 27
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have a bucket list? No."
            },
            {
              "qas": [
                {
                  "question": "What's on your bucket list?",
                  "id": "id86",
                  "answers": [
                    {
                      "text": "Nothing!",
                      "answer_start": 28
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's on your bucket list? Nothing!"
            },
            {
              "qas": [
                {
                  "question": "You're drawing more on your economics knowledge than music?",
                  "id": "id87",
                  "answers": [
                    {
                      "text": "Yes, because even if I love music, I don't love the music industry.",
                      "answer_start": 60
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "You're drawing more on your economics knowledge than music? Yes, because even if I love music, I don't love the music industry."
            },
            {
              "qas": [
                {
                  "question": "Is your job related to what you studied?",
                  "id": "id88",
                  "answers": [
                    {
                      "text": "A little bit. My economics minor is helping me with that a lot.",
                      "answer_start": 41
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is your job related to what you studied? A little bit. My economics minor is helping me with that a lot."
            },
            {
              "qas": [
                {
                  "question": "Yeah it really sucks. So do you think you're prepared for like the real world after NYU?",
                  "id": "id89",
                  "answers": [
                    {
                      "text": "I feel that I'm definitely more prepared than an average person of my age.",
                      "answer_start": 89
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Yeah it really sucks. So do you think you're prepared for like the real world after NYU? I feel that I'm definitely more prepared than an average person of my age."
            },
            {
              "qas": [
                {
                  "question": "Do most people stay in Abu Dhabi or the UAE or they just go around the world?",
                  "id": "id90",
                  "answers": [
                    {
                      "text": "More people go to another country after graduation, but a considerable number stay in the UAE.",
                      "answer_start": 78
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do most people stay in Abu Dhabi or the UAE or they just go around the world? More people go to another country after graduation, but a considerable number stay in the UAE."
            },
            {
              "qas": [
                {
                  "question": "Is this robot artificial intelligence connected to the work, right?",
                  "id": "id91",
                  "answers": [
                    {
                      "text": "Nope. It's just something I'm doing over summer.",
                      "answer_start": 68
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is this robot artificial intelligence connected to the work, right? Nope. It's just something I'm doing over summer."
            },
            {
              "qas": [
                {
                  "question": "What is the vision of NYUAD?",
                  "id": "id92",
                  "answers": [
                    {
                      "text": "Summed up, NYUAD's vision is to be the new paradigm in higher education and a magnet for diverse and creative people from around the world.",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the vision of NYUAD? Summed up, NYUAD's vision is to be the new paradigm in higher education and a magnet for diverse and creative people from around the world."
            },
            {
              "qas": [
                {
                  "question": "When you start your studies is there any way that you learn what possibility you have to find a job if you follow this direction that you like mostly?",
                  "id": "id93",
                  "answers": [
                    {
                      "text": "There are events in which you can speak to alumni and there is Career Development Center, which is a department that helps you plan your career. But finding a job is on you.",
                      "answer_start": 151
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "When you start your studies is there any way that you learn what possibility you have to find a job if you follow this direction that you like mostly? There are events in which you can speak to alumni and there is Career Development Center, which is a department that helps you plan your career. But finding a job is on you."
            },
            {
              "qas": [
                {
                  "question": "What is the percentage of people that go into work straight away from the degree?",
                  "id": "id94",
                  "answers": [
                    {
                      "text": "There are some statistics, but it could be that the people who take those surveys are the ones who got the job. It's more than 90% though.",
                      "answer_start": 82
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the percentage of people that go into work straight away from the degree? There are some statistics, but it could be that the people who take those surveys are the ones who got the job. It's more than 90% though."
            },
            {
              "qas": [
                {
                  "question": "Have you graduated? You graduated, right?",
                  "id": "id95",
                  "answers": [
                    {
                      "text": "I graduated in May 2019.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you graduated? You graduated, right? I graduated in May 2019."
            },
            {
              "qas": [
                {
                  "question": "Hi",
                  "id": "id96",
                  "answers": [
                    {
                      "text": "Hello!",
                      "answer_start": 3
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Hi Hello!"
            },
            {
              "qas": [
                {
                  "question": "Why the red?",
                  "id": "id97",
                  "answers": [
                    {
                      "text": "I started dying my hair red since I was about 15. I just like that it stands out, since I like attention.",
                      "answer_start": 13
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Why the red? I started dying my hair red since I was about 15. I just like that it stands out, since I like attention."
            },
            {
              "qas": [
                {
                  "question": "How does NYUAD support students in adjusting to life in the UAE?",
                  "id": "id98",
                  "answers": [
                    {
                      "text": "There's counseling services and many events to connect and talk about your experiences.",
                      "answer_start": 65
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How does NYUAD support students in adjusting to life in the UAE? There's counseling services and many events to connect and talk about your experiences."
            },
            {
              "qas": [
                {
                  "question": "How does NYUAD support students with their post graduation careers while in their final year?",
                  "id": "id99",
                  "answers": [
                    {
                      "text": "There are events in which you can speak to alumni and there is Career Development Center, which is a department that helps you out with planning your career. But finding a job is on you.",
                      "answer_start": 94
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How does NYUAD support students with their post graduation careers while in their final year? There are events in which you can speak to alumni and there is Career Development Center, which is a department that helps you out with planning your career. But finding a job is on you."
            },
            {
              "qas": [
                {
                  "question": "Did you think that after completing your studies that this whole idea of leadership has an affect to change the character of the person that enters this university? Because it is a whole way of thinking. It's not just that one course. So at the end, how much do the students finally, when they graduate, changed and accepted this as a role?",
                  "id": "id100",
                  "answers": [
                    {
                      "text": "I do think I'm a better leader now. NYUAD made me so much more open-minded and goal-oriented, and I have better social skills, too.",
                      "answer_start": 341
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you think that after completing your studies that this whole idea of leadership has an affect to change the character of the person that enters this university? Because it is a whole way of thinking. It's not just that one course. So at the end, how much do the students finally, when they graduate, changed and accepted this as a role? I do think I'm a better leader now. NYUAD made me so much more open-minded and goal-oriented, and I have better social skills, too."
            },
            {
              "qas": [
                {
                  "question": "What is it like to be part of such a diverse community?",
                  "id": "id101",
                  "answers": [
                    {
                      "text": "It makes you more tolerant and you understand the world better.",
                      "answer_start": 56
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is it like to be part of such a diverse community? It makes you more tolerant and you understand the world better."
            },
            {
              "qas": [
                {
                  "question": "Where are most NYUAD students from?",
                  "id": "id102",
                  "answers": [
                    {
                      "text": "Probably only 15 percent are Emirati students, and everyone else is from around the world.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where are most NYUAD students from? Probably only 15 percent are Emirati students, and everyone else is from around the world."
            },
            {
              "qas": [
                {
                  "question": "Did you have courses on the cultural awareness?",
                  "id": "id103",
                  "answers": [
                    {
                      "text": "There are such courses, if you want to take them, but with so much diversity all around you, you kind of learn by yourself.",
                      "answer_start": 48
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you have courses on the cultural awareness? There are such courses, if you want to take them, but with so much diversity all around you, you kind of learn by yourself."
            },
            {
              "qas": [
                {
                  "question": "Do you have International Student's Day? Where students from all over the country represent their culture and so on?",
                  "id": "id104",
                  "answers": [
                    {
                      "text": "We don't really have an International Students Day because every day is International Students Day here.",
                      "answer_start": 117
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have International Student's Day? Where students from all over the country represent their culture and so on? We don't really have an International Students Day because every day is International Students Day here."
            },
            {
              "qas": [
                {
                  "question": "From where will students come, and how many?",
                  "id": "id105",
                  "answers": [
                    {
                      "text": "Probably only 15 percent of Emirati students, and everyone else is from around the world.",
                      "answer_start": 45
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "From where will students come, and how many? Probably only 15 percent of Emirati students, and everyone else is from around the world."
            },
            {
              "qas": [
                {
                  "question": "How old are you?",
                  "id": "id106",
                  "answers": [
                    {
                      "text": "I am 22.",
                      "answer_start": 17
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How old are you? I am 22."
            },
            {
              "qas": [
                {
                  "question": "Are you a student here?",
                  "id": "id107",
                  "answers": [
                    {
                      "text": "I just graduated from New York University Abu Dhabi!",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are you a student here? I just graduated from New York University Abu Dhabi!"
            },
            {
              "qas": [
                {
                  "question": "What is your name?",
                  "id": "id108",
                  "answers": [
                    {
                      "text": "My name is Margarita.",
                      "answer_start": 19
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your name? My name is Margarita."
            },
            {
              "qas": [
                {
                  "question": "Do you have siblings who study in New York or Abu Dhabi?",
                  "id": "id109",
                  "answers": [
                    {
                      "text": "I don't.",
                      "answer_start": 57
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have siblings who study in New York or Abu Dhabi? I don't."
            },
            {
              "qas": [
                {
                  "question": "Will I need to learn Arabic to get by in Abu Dhabi and the UAE?",
                  "id": "id110",
                  "answers": [
                    {
                      "text": "All classes are in English and it is not mandatory to take any language classes.",
                      "answer_start": 64
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Will I need to learn Arabic to get by in Abu Dhabi and the UAE? All classes are in English and it is not mandatory to take any language classes."
            },
            {
              "qas": [
                {
                  "question": "Those are the two main languages there.",
                  "id": "id111",
                  "answers": [
                    {
                      "text": "Exactly.",
                      "answer_start": 40
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Those are the two main languages there. Exactly."
            },
            {
              "qas": [
                {
                  "question": "Have you always spoken English?",
                  "id": "id112",
                  "answers": [
                    {
                      "text": "I actually learned English from listening to Western music! And then I started having English classes from 5th grade on in school.",
                      "answer_start": 32
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you always spoken English? I actually learned English from listening to Western music! And then I started having English classes from 5th grade on in school."
            },
            {
              "qas": [
                {
                  "question": "What about languages? How many languages do you speak?",
                  "id": "id113",
                  "answers": [
                    {
                      "text": "I speak five languages: Romanian, Russian, French, English and Spanish.",
                      "answer_start": 55
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What about languages? How many languages do you speak? I speak five languages: Romanian, Russian, French, English and Spanish."
            },
            {
              "qas": [
                {
                  "question": "Do you speak Russian?",
                  "id": "id114",
                  "answers": [
                    {
                      "text": "I speak Romanian, Russian, French, English and Spanish.",
                      "answer_start": 22
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you speak Russian? I speak Romanian, Russian, French, English and Spanish."
            },
            {
              "qas": [
                {
                  "question": "How did you learn Spanish?",
                  "id": "id115",
                  "answers": [
                    {
                      "text": "I studied abroad in Buenos Aires and since Spanish is quite close to the Romanian roots, Latin, I learned it in that semester.",
                      "answer_start": 27
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How did you learn Spanish? I studied abroad in Buenos Aires and since Spanish is quite close to the Romanian roots, Latin, I learned it in that semester."
            },
            {
              "qas": [
                {
                  "question": "How come Moldovans speak both of those languages?",
                  "id": "id116",
                  "answers": [
                    {
                      "text": "Moldova was originally a region of Romania, and then we became a part of the Russian Empire and then the Soviet Union, and so after we became independent we were left with two languages.",
                      "answer_start": 50
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How come Moldovans speak both of those languages? Moldova was originally a region of Romania, and then we became a part of the Russian Empire and then the Soviet Union, and so after we became independent we were left with two languages."
            },
            {
              "qas": [
                {
                  "question": "What is the national language of Moldova?",
                  "id": "id117",
                  "answers": [
                    {
                      "text": "Moldova's national language is Romanian, but everyone also speaks Russian.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the national language of Moldova? Moldova's national language is Romanian, but everyone also speaks Russian."
            },
            {
              "qas": [
                {
                  "question": "What are the languages spoken in Abu Dhabi?",
                  "id": "id118",
                  "answers": [
                    {
                      "text": "Most people know English, but Arabic, Persian, Hindi, and Urdu are also spoken.",
                      "answer_start": 44
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are the languages spoken in Abu Dhabi? Most people know English, but Arabic, Persian, Hindi, and Urdu are also spoken."
            },
            {
              "qas": [
                {
                  "question": "What language do your grandparents speak?",
                  "id": "id119",
                  "answers": [
                    {
                      "text": "My dad's parents speak Romanian, but my mom's parents speak in Ukrainian to me.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What language do your grandparents speak? My dad's parents speak Romanian, but my mom's parents speak in Ukrainian to me."
            },
            {
              "qas": [
                {
                  "question": "What language do you speak in your family?",
                  "id": "id120",
                  "answers": [
                    {
                      "text": "My family speaks both languages! My mom's first language is Russian and my dad's is Romanian, so it's a nice mix.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What language do you speak in your family? My family speaks both languages! My mom's first language is Russian and my dad's is Romanian, so it's a nice mix."
            },
            {
              "qas": [
                {
                  "question": "Have You learned Arabic?",
                  "id": "id121",
                  "answers": [
                    {
                      "text": "No, I didn't learn Arabic. But I learned Spanish instead!",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have You learned Arabic? No, I didn't learn Arabic. But I learned Spanish instead!"
            },
            {
              "qas": [
                {
                  "question": "Is it true that for example - I guess this would apply to Romanians or maybe. But is it true that you guys can understand Italian but Italians can't understand you?",
                  "id": "id122",
                  "answers": [
                    {
                      "text": "Sometimes! They're still different languages, but I can kind of get a feeling of what they're talking about.",
                      "answer_start": 165
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is it true that for example - I guess this would apply to Romanians or maybe. But is it true that you guys can understand Italian but Italians can't understand you? Sometimes! They're still different languages, but I can kind of get a feeling of what they're talking about."
            },
            {
              "qas": [
                {
                  "question": "What languages do you speak in school?",
                  "id": "id123",
                  "answers": [
                    {
                      "text": "We have more Romanian than Russian schools, but we do have both.",
                      "answer_start": 39
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What languages do you speak in school? We have more Romanian than Russian schools, but we do have both."
            },
            {
              "qas": [
                {
                  "question": "How come you speak so many languages?",
                  "id": "id124",
                  "answers": [
                    {
                      "text": "Well everyone in Romania speaks... aah I'm from Moldova. Everyone in Moldova speaks Romanian and Russian, and everything else I learned at university or in school.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How come you speak so many languages? Well everyone in Romania speaks... aah I'm from Moldova. Everyone in Moldova speaks Romanian and Russian, and everything else I learned at university or in school."
            },
            {
              "qas": [
                {
                  "question": "Where did you learn these languages?",
                  "id": "id125",
                  "answers": [
                    {
                      "text": "Well everyone in Moldova speaks Romanian and Russian, and the rest of the languages I learned at school or university.",
                      "answer_start": 37
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where did you learn these languages? Well everyone in Moldova speaks Romanian and Russian, and the rest of the languages I learned at school or university."
            },
            {
              "qas": [
                {
                  "question": "And how important is to have leadership skills? I mean you have to be interested in leading? I mean because you mentioned previously that having leadership skills is something that counts. And if you're not interested, if you're very very good for example only as a scientist, is that to critical to be accepted?",
                  "id": "id126",
                  "answers": [
                    {
                      "text": "I think this university really tries to get leaders on board, and leaders in different types of ways. People who believe in the bigger picture and try to make a change.",
                      "answer_start": 313
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "And how important is to have leadership skills? I mean you have to be interested in leading? I mean because you mentioned previously that having leadership skills is something that counts. And if you're not interested, if you're very very good for example only as a scientist, is that to critical to be accepted? I think this university really tries to get leaders on board, and leaders in different types of ways. People who believe in the bigger picture and try to make a change."
            },
            {
              "qas": [
                {
                  "question": "Do you have a boyfriend?",
                  "id": "id127",
                  "answers": [
                    {
                      "text": "I like to keep things like this private.",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have a boyfriend? I like to keep things like this private."
            },
            {
              "qas": [
                {
                  "question": "What is a relationship deal breaker for you?",
                  "id": "id128",
                  "answers": [
                    {
                      "text": "If they're not willing to communicate and solve problems effectively, if they're not nice to other people, if they're not ambitious.",
                      "answer_start": 45
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is a relationship deal breaker for you? If they're not willing to communicate and solve problems effectively, if they're not nice to other people, if they're not ambitious."
            },
            {
              "qas": [
                {
                  "question": "Who was your first crush?",
                  "id": "id129",
                  "answers": [
                    {
                      "text": "My first crush was a boy at my kindergarten, but there's no interesting story about that.",
                      "answer_start": 26
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Who was your first crush? My first crush was a boy at my kindergarten, but there's no interesting story about that."
            },
            {
              "qas": [
                {
                  "question": "Have you ever had a secret admirer?",
                  "id": "id130",
                  "answers": [
                    {
                      "text": "Oh my god, yes! Someone once wrote anonymously a long post in our university group designated for compliments, and I'm still not sure who that was.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you ever had a secret admirer? Oh my god, yes! Someone once wrote anonymously a long post in our university group designated for compliments, and I'm still not sure who that was."
            },
            {
              "qas": [
                {
                  "question": "Yes interesting. You're also taking economics. Is that another interest of yours? Or ...?",
                  "id": "id131",
                  "answers": [
                    {
                      "text": "I didn't double major because I was a bit behind on credits for my economics major and I was supposed to take a summer class. But I also had an internship opportunity in New York and I had to choose. So I chose the internship in New York over the major.",
                      "answer_start": 90
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Yes interesting. You're also taking economics. Is that another interest of yours? Or ...? I didn't double major because I was a bit behind on credits for my economics major and I was supposed to take a summer class. But I also had an internship opportunity in New York and I had to choose. So I chose the internship in New York over the major."
            },
            {
              "qas": [
                {
                  "question": "Did you double major?",
                  "id": "id132",
                  "answers": [
                    {
                      "text": "I didn't double major.",
                      "answer_start": 22
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you double major? I didn't double major."
            },
            {
              "qas": [
                {
                  "question": "What do you study?",
                  "id": "id133",
                  "answers": [
                    {
                      "text": "I studied music and economics. I'm a music major, economics minor and in music I do mostly composition and sound engineering.",
                      "answer_start": 19
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What do you study? I studied music and economics. I'm a music major, economics minor and in music I do mostly composition and sound engineering."
            },
            {
              "qas": [
                {
                  "question": "Oh, great. So which one do you like better? Or which one is like the most appealing to you?",
                  "id": "id134",
                  "answers": [
                    {
                      "text": "Music completes this artistic and techie side of me because I studied composition and sound engineering, but economics fulfilled my social and analytical side, because I wanted to learn about how to make an impact in the world and I love math.",
                      "answer_start": 92
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Oh, great. So which one do you like better? Or which one is like the most appealing to you? Music completes this artistic and techie side of me because I studied composition and sound engineering, but economics fulfilled my social and analytical side, because I wanted to learn about how to make an impact in the world and I love math."
            },
            {
              "qas": [
                {
                  "question": "What degrees does NYUAD offer?",
                  "id": "id135",
                  "answers": [
                    {
                      "text": "NYUAD offers Bachelor of Science and Arts.",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What degrees does NYUAD offer? NYUAD offers Bachelor of Science and Arts."
            },
            {
              "qas": [
                {
                  "question": "Can I major in Engineering at NYUAD?",
                  "id": "id136",
                  "answers": [
                    {
                      "text": "Of course you can!",
                      "answer_start": 37
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Can I major in Engineering at NYUAD? Of course you can!"
            },
            {
              "qas": [
                {
                  "question": "You have possibility and the freedom to create perhaps like a tuned major from two different schools to combine, so to have a different specialty on something?",
                  "id": "id137",
                  "answers": [
                    {
                      "text": "You can do multiple majors.",
                      "answer_start": 160
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "You have possibility and the freedom to create perhaps like a tuned major from two different schools to combine, so to have a different specialty on something? You can do multiple majors."
            },
            {
              "qas": [
                {
                  "question": "When do I have to declare my major?",
                  "id": "id138",
                  "answers": [
                    {
                      "text": "You have to declare your major at the end of the second year.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "When do I have to declare my major? You have to declare your major at the end of the second year."
            },
            {
              "qas": [
                {
                  "question": "Is there a system that in first year you do some courses and after you decide what you actually want to do?",
                  "id": "id139",
                  "answers": [
                    {
                      "text": "You have to declare your major at the end of the second year. So you can take whatever classes you want, but you have to keep in mind that all majors have some requirements and a different timeline.",
                      "answer_start": 108
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is there a system that in first year you do some courses and after you decide what you actually want to do? You have to declare your major at the end of the second year. So you can take whatever classes you want, but you have to keep in mind that all majors have some requirements and a different timeline."
            },
            {
              "qas": [
                {
                  "question": "Is the university well advertised abroad?",
                  "id": "id140",
                  "answers": [
                    {
                      "text": "Of course NYUAD is trying to be known everywhere, but it's so only so far that you can reach.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is the university well advertised abroad? Of course NYUAD is trying to be known everywhere, but it's so only so far that you can reach."
            },
            {
              "qas": [
                {
                  "question": "You studied in Moldova for a long time. How about the teachers. How does it feel with them there and how does it feel, like, studying in Moldova? Can you compare it?",
                  "id": "id141",
                  "answers": [
                    {
                      "text": "Here I feel professors are more open-minded and qualified than back at home. But they're also paid better.",
                      "answer_start": 166
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "You studied in Moldova for a long time. How about the teachers. How does it feel with them there and how does it feel, like, studying in Moldova? Can you compare it? Here I feel professors are more open-minded and qualified than back at home. But they're also paid better."
            },
            {
              "qas": [
                {
                  "question": "How's like in Moldova in terms of family? Like do you have a lot of cousins?",
                  "id": "id142",
                  "answers": [
                    {
                      "text": "I feel like the previous generations would have way more kids. I have some cousins, not that many because they're all older than me... But the families are quite conservative still.",
                      "answer_start": 77
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How's like in Moldova in terms of family? Like do you have a lot of cousins? I feel like the previous generations would have way more kids. I have some cousins, not that many because they're all older than me... But the families are quite conservative still."
            },
            {
              "qas": [
                {
                  "question": "Do you keep in touch with your old friends?",
                  "id": "id143",
                  "answers": [
                    {
                      "text": "I skype my old friends a lot.",
                      "answer_start": 44
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you keep in touch with your old friends? I skype my old friends a lot."
            },
            {
              "qas": [
                {
                  "question": "How often do you skype your family?",
                  "id": "id144",
                  "answers": [
                    {
                      "text": "I skype with my family once a week, and I never skip a week.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How often do you skype your family? I skype with my family once a week, and I never skip a week."
            },
            {
              "qas": [
                {
                  "question": "Where are you from?",
                  "id": "id145",
                  "answers": [
                    {
                      "text": "I'm from Moldova, a small country in Eastern Europe.",
                      "answer_start": 20
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where are you from? I'm from Moldova, a small country in Eastern Europe."
            },
            {
              "qas": [
                {
                  "question": "Do you miss home a lot?",
                  "id": "id146",
                  "answers": [
                    {
                      "text": "I've lived abroad before so I'm used to being away, so no, I don't miss home too much. Plus, I skype my family often.",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you miss home a lot? I've lived abroad before so I'm used to being away, so no, I don't miss home too much. Plus, I skype my family often."
            },
            {
              "qas": [
                {
                  "question": "How is the weather like in Moldova?",
                  "id": "id147",
                  "answers": [
                    {
                      "text": "In Moldova we have all types of weather. We have four seasons and it gets from minus 20 degrees Celsius to 40. So we have a huge range and all four seasons - Summer, Fall, Winter, Spring. So I miss that.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How is the weather like in Moldova? In Moldova we have all types of weather. We have four seasons and it gets from minus 20 degrees Celsius to 40. So we have a huge range and all four seasons - Summer, Fall, Winter, Spring. So I miss that."
            },
            {
              "qas": [
                {
                  "question": "What does Moldova border?",
                  "id": "id148",
                  "answers": [
                    {
                      "text": "Moldova is landlocked between Romania and Ukraine.",
                      "answer_start": 26
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What does Moldova border? Moldova is landlocked between Romania and Ukraine."
            },
            {
              "qas": [
                {
                  "question": "What was the reason for you not to stay in your country or want to go out?",
                  "id": "id149",
                  "answers": [
                    {
                      "text": "Moldova is the poorest country in Europe by GDP per capita and there are limited opportunities. It's not an environment that I wanted to be in.",
                      "answer_start": 75
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was the reason for you not to stay in your country or want to go out? Moldova is the poorest country in Europe by GDP per capita and there are limited opportunities. It's not an environment that I wanted to be in."
            },
            {
              "qas": [
                {
                  "question": "What's the cuisine like in Moldova?",
                  "id": "id150",
                  "answers": [
                    {
                      "text": "Moldova used to be part of Romania and then Soviet Union. So the foods are combined from all over the place. We have Romanian food. We have Russian food. We eat a lot of soups, meat, rice, salads, and Western food of course.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's the cuisine like in Moldova? Moldova used to be part of Romania and then Soviet Union. So the foods are combined from all over the place. We have Romanian food. We have Russian food. We eat a lot of soups, meat, rice, salads, and Western food of course."
            },
            {
              "qas": [
                {
                  "question": "What is Moldovan national music like?",
                  "id": "id151",
                  "answers": [
                    {
                      "text": "Moldovan national music is either really sad or really happy, has a lot of brass and is closely similar to Romanian music actually.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is Moldovan national music like? Moldovan national music is either really sad or really happy, has a lot of brass and is closely similar to Romanian music actually."
            },
            {
              "qas": [
                {
                  "question": "OK! If you could describe your country in a few short sentences. What would you say?",
                  "id": "id152",
                  "answers": [
                    {
                      "text": "So Moldova has a great climate. We have four seasons. We speak two languages from the very get go. And we have really hot girls.",
                      "answer_start": 85
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "OK! If you could describe your country in a few short sentences. What would you say? So Moldova has a great climate. We have four seasons. We speak two languages from the very get go. And we have really hot girls."
            },
            {
              "qas": [
                {
                  "question": "When did you first know that you want to work with music?",
                  "id": "id153",
                  "answers": [
                    {
                      "text": "Back in kindergarten, I would always sing to my dolls and attempt to write music, so I always thought I'd be a singer.",
                      "answer_start": 58
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "When did you first know that you want to work with music? Back in kindergarten, I would always sing to my dolls and attempt to write music, so I always thought I'd be a singer."
            },
            {
              "qas": [
                {
                  "question": "What artist is your biggest inspiration?",
                  "id": "id154",
                  "answers": [
                    {
                      "text": "Definitely Pentatonix.",
                      "answer_start": 41
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What artist is your biggest inspiration? Definitely Pentatonix."
            },
            {
              "qas": [
                {
                  "question": "How often do you compose?",
                  "id": "id155",
                  "answers": [
                    {
                      "text": "I compose whenever I have the time, necessity or inspiration.",
                      "answer_start": 26
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How often do you compose? I compose whenever I have the time, necessity or inspiration."
            },
            {
              "qas": [
                {
                  "question": "Did you go to a musical school?",
                  "id": "id156",
                  "answers": [
                    {
                      "text": "I did when I was in Moldova, but not officially, so I don't have a diploma.",
                      "answer_start": 32
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you go to a musical school? I did when I was in Moldova, but not officially, so I don't have a diploma."
            },
            {
              "qas": [
                {
                  "question": "Do you compose your own piano songs?",
                  "id": "id157",
                  "answers": [
                    {
                      "text": "I do, but unfortunately I don't have a piano with me right now! So you'll just have to go to margaritabee.com",
                      "answer_start": 37
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you compose your own piano songs? I do, but unfortunately I don't have a piano with me right now! So you'll just have to go to margaritabee.com"
            },
            {
              "qas": [
                {
                  "question": "Do you want to be a DJ in the future?",
                  "id": "id158",
                  "answers": [
                    {
                      "text": "I don't think so, but I find it really fun.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you want to be a DJ in the future? I don't think so, but I find it really fun."
            },
            {
              "qas": [
                {
                  "question": "How is it here music in like this region?",
                  "id": "id159",
                  "answers": [
                    {
                      "text": "I find it beautiful, although I haven't learned much about it.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How is it here music in like this region? I find it beautiful, although I haven't learned much about it."
            },
            {
              "qas": [
                {
                  "question": "What languages do you like writing songs in, though?",
                  "id": "id160",
                  "answers": [
                    {
                      "text": "I find it easier to write songs in English and Romanian, but I don't mind the other languages either.",
                      "answer_start": 53
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What languages do you like writing songs in, though? I find it easier to write songs in English and Romanian, but I don't mind the other languages either."
            },
            {
              "qas": [
                {
                  "question": "What is your composition style?",
                  "id": "id161",
                  "answers": [
                    {
                      "text": "I like writing epic or dramatic instrumental music, but also lyrics for pop songs.",
                      "answer_start": 32
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your composition style? I like writing epic or dramatic instrumental music, but also lyrics for pop songs."
            },
            {
              "qas": [
                {
                  "question": "What is your favorite genre?",
                  "id": "id162",
                  "answers": [
                    {
                      "text": "I love all music genres, really! I especially like dubstep, pop, hip hop and classical sometimes.",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your favorite genre? I love all music genres, really! I especially like dubstep, pop, hip hop and classical sometimes."
            },
            {
              "qas": [
                {
                  "question": "What do you like most about being a musician?",
                  "id": "id163",
                  "answers": [
                    {
                      "text": "I love the emotions that come with music and being able to connect to other people through music.",
                      "answer_start": 46
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What do you like most about being a musician? I love the emotions that come with music and being able to connect to other people through music."
            },
            {
              "qas": [
                {
                  "question": "Do you play any instruments?",
                  "id": "id164",
                  "answers": [
                    {
                      "text": "I play the piano and a little bit of ukulele and cello and I've been playing flute for a year. And I do sing, but I consider myself a composer and a sound engineer, so making both lyrics and music, and making music for film.",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you play any instruments? I play the piano and a little bit of ukulele and cello and I've been playing flute for a year. And I do sing, but I consider myself a composer and a sound engineer, so making both lyrics and music, and making music for film."
            },
            {
              "qas": [
                {
                  "question": "What was the last song that made you dance?",
                  "id": "id165",
                  "answers": [
                    {
                      "text": "I really can't remember!",
                      "answer_start": 44
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was the last song that made you dance? I really can't remember!"
            },
            {
              "qas": [
                {
                  "question": "Are you good at playing the cello?",
                  "id": "id166",
                  "answers": [
                    {
                      "text": "I've played it for less than a year, so I'm probably not that good.",
                      "answer_start": 35
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are you good at playing the cello? I've played it for less than a year, so I'm probably not that good."
            },
            {
              "qas": [
                {
                  "question": "How do you handle mistakes during a performance?",
                  "id": "id167",
                  "answers": [
                    {
                      "text": "If possible, I pretend like nothing happened and go on, but if the mistake was way too obvious, I can make a joke about it.",
                      "answer_start": 49
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do you handle mistakes during a performance? If possible, I pretend like nothing happened and go on, but if the mistake was way too obvious, I can make a joke about it."
            },
            {
              "qas": [
                {
                  "question": "What are you interested in that most people haven't heard of?",
                  "id": "id168",
                  "answers": [
                    {
                      "text": "Mixing and mastering, which basically means editing music.",
                      "answer_start": 62
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are you interested in that most people haven't heard of? Mixing and mastering, which basically means editing music."
            },
            {
              "qas": [
                {
                  "question": "Would you say you discovered electronic music production through your brother or was it something on your own?",
                  "id": "id169",
                  "answers": [
                    {
                      "text": "My family doesn't actually like music as heavily as I do. I always loved music and it became my unique thing.",
                      "answer_start": 111
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Would you say you discovered electronic music production through your brother or was it something on your own? My family doesn't actually like music as heavily as I do. I always loved music and it became my unique thing."
            },
            {
              "qas": [
                {
                  "question": "What was the first song you've learned?",
                  "id": "id170",
                  "answers": [
                    {
                      "text": "My first songs were Moldovan national songs and some Russian pop songs, but the first song in English that I fell in love with was Beyonce's Irreplaceable when I was 6.",
                      "answer_start": 40
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was the first song you've learned? My first songs were Moldovan national songs and some Russian pop songs, but the first song in English that I fell in love with was Beyonce's Irreplaceable when I was 6."
            },
            {
              "qas": [
                {
                  "question": "What are your biggest music influences?",
                  "id": "id171",
                  "answers": [
                    {
                      "text": "My music has been influenced by Western pop, classical and electronic music.",
                      "answer_start": 40
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are your biggest music influences? My music has been influenced by Western pop, classical and electronic music."
            },
            {
              "qas": [
                {
                  "question": "Have you ever performed in the street?",
                  "id": "id172",
                  "answers": [
                    {
                      "text": "Of course! I've done it a couple of times with my guitarist friends - I was singing. It was great, but we barely got any money for it, but I was very happy to just be there and enjoy making music.",
                      "answer_start": 39
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you ever performed in the street? Of course! I've done it a couple of times with my guitarist friends - I was singing. It was great, but we barely got any money for it, but I was very happy to just be there and enjoy making music."
            },
            {
              "qas": [
                {
                  "question": "When was the last time you learned to play an instrument?",
                  "id": "id173",
                  "answers": [
                    {
                      "text": "Probably flute!",
                      "answer_start": 58
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "When was the last time you learned to play an instrument? Probably flute!"
            },
            {
              "qas": [
                {
                  "question": "How often do you practice music?",
                  "id": "id174",
                  "answers": [
                    {
                      "text": "Right now not as often as I'd like. Maybe once every few days?",
                      "answer_start": 33
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How often do you practice music? Right now not as often as I'd like. Maybe once every few days?"
            },
            {
              "qas": [
                {
                  "question": "How do you stay motivated when you're not satisfied with your own work?",
                  "id": "id175",
                  "answers": [
                    {
                      "text": "Show your music to the world and you're likely to get some positive feedback to keep you going and some constructive criticism to improve your work!",
                      "answer_start": 72
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do you stay motivated when you're not satisfied with your own work? Show your music to the world and you're likely to get some positive feedback to keep you going and some constructive criticism to improve your work!"
            },
            {
              "qas": [
                {
                  "question": "Can you sing something of your own?",
                  "id": "id176",
                  "answers": [
                    {
                      "text": "Sure! Visit my website: margaritabee.com",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Can you sing something of your own? Sure! Visit my website: margaritabee.com"
            },
            {
              "qas": [
                {
                  "question": "What purpose do you want your music to serve?",
                  "id": "id177",
                  "answers": [
                    {
                      "text": "To make the world more beautiful and to make people feel something.",
                      "answer_start": 46
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What purpose do you want your music to serve? To make the world more beautiful and to make people feel something."
            },
            {
              "qas": [
                {
                  "question": "What do you need to succeed in music?",
                  "id": "id178",
                  "answers": [
                    {
                      "text": "To succeed in music, aside from talent, you need to have people skills and know a few things about the music industry.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What do you need to succeed in music? To succeed in music, aside from talent, you need to have people skills and know a few things about the music industry."
            },
            {
              "qas": [
                {
                  "question": "What's your favorite thing to do at the university?",
                  "id": "id179",
                  "answers": [
                    {
                      "text": "Ummm I like singing, doing sports, reading, hanging out with friends and traveling",
                      "answer_start": 52
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your favorite thing to do at the university? Ummm I like singing, doing sports, reading, hanging out with friends and traveling"
            },
            {
              "qas": [
                {
                  "question": "Did you have music classes in your school?",
                  "id": "id180",
                  "answers": [
                    {
                      "text": "We had one music class in school, but it wasn't extensive.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you have music classes in your school? We had one music class in school, but it wasn't extensive."
            },
            {
              "qas": [
                {
                  "question": "How long have you been playing piano for?",
                  "id": "id181",
                  "answers": [
                    {
                      "text": "Well I've been playing piano for the last 14 years so I must be okay at it!",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How long have you been playing piano for? Well I've been playing piano for the last 14 years so I must be okay at it!"
            },
            {
              "qas": [
                {
                  "question": "How do you write music?",
                  "id": "id182",
                  "answers": [
                    {
                      "text": "When I compose I sit down at the piano and sing random things that pop in my mind and start off with the melody, after which I add lyrics. Or I open a software that will allow me to hear what I'm working on and again, try random things.",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do you write music? When I compose I sit down at the piano and sing random things that pop in my mind and start off with the melody, after which I add lyrics. Or I open a software that will allow me to hear what I'm working on and again, try random things."
            },
            {
              "qas": [
                {
                  "question": "Can you dance?",
                  "id": "id183",
                  "answers": [
                    {
                      "text": "Yes! WATCH ME (dances)",
                      "answer_start": 15
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Can you dance? Yes! WATCH ME (dances)"
            },
            {
              "qas": [
                {
                  "question": "How long have you been playing the cello for?",
                  "id": "id184",
                  "answers": [
                    {
                      "text": "I've played the cello for less than a year, so I'm probably not that good.",
                      "answer_start": 46
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How long have you been playing the cello for? I've played the cello for less than a year, so I'm probably not that good."
            },
            {
              "qas": [
                {
                  "question": "How long have you been playing the ukulele for?",
                  "id": "id185",
                  "answers": [
                    {
                      "text": "I've played the ukulele for less than a year, so I'm probably not that good.",
                      "answer_start": 48
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How long have you been playing the ukulele for? I've played the ukulele for less than a year, so I'm probably not that good."
            },
            {
              "qas": [
                {
                  "question": "How long have you been playing the flute for?",
                  "id": "id186",
                  "answers": [
                    {
                      "text": "I've played the flute for less than a year, so I'm probably not that good.",
                      "answer_start": 46
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How long have you been playing the flute for? I've played the flute for less than a year, so I'm probably not that good."
            },
            {
              "qas": [
                {
                  "question": "How do you go about composing?",
                  "id": "id187",
                  "answers": [
                    {
                      "text": "When composing music, I sit down at the piano and sing random things that come to my mind and start off with the melody, after which I add lyrics. Or I open a software that will allow me to hear what I work on and again, try random things.",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do you go about composing? When composing music, I sit down at the piano and sing random things that come to my mind and start off with the melody, after which I add lyrics. Or I open a software that will allow me to hear what I work on and again, try random things."
            },
            {
              "qas": [
                {
                  "question": "What do you do if you get something wrong on stage?",
                  "id": "id188",
                  "answers": [
                    {
                      "text": "If possible, I pretend nothing happened and go on, but if the mistake was way too obvious, I can make a joke about it and go on.",
                      "answer_start": 52
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What do you do if you get something wrong on stage? If possible, I pretend nothing happened and go on, but if the mistake was way too obvious, I can make a joke about it and go on."
            },
            {
              "qas": [
                {
                  "question": "Have you ever sung in the street?",
                  "id": "id189",
                  "answers": [
                    {
                      "text": "Of course! I've done it a couple of times with my guitarist friends - I was singing. It was great. We barely got any money for it, but I was happy to just be there and make music.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you ever sung in the street? Of course! I've done it a couple of times with my guitarist friends - I was singing. It was great. We barely got any money for it, but I was happy to just be there and make music."
            },
            {
              "qas": [
                {
                  "question": "What do you love the most about your field?",
                  "id": "id190",
                  "answers": [
                    {
                      "text": "I love that it always carries emotions with it. Whether it's anger, happiness, sadness, love, it makes you feel something, and I love feeling things.",
                      "answer_start": 44
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What do you love the most about your field? I love that it always carries emotions with it. Whether it's anger, happiness, sadness, love, it makes you feel something, and I love feeling things."
            },
            {
              "qas": [
                {
                  "question": "What was you capstone about?",
                  "id": "id191",
                  "answers": [
                    {
                      "text": "I took a Romanian folk song and created two versions of it mixed with Dubstep!",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was you capstone about? I took a Romanian folk song and created two versions of it mixed with Dubstep!"
            },
            {
              "qas": [
                {
                  "question": "What are the craziest myths that you've heard before coming here about this country?",
                  "id": "id192",
                  "answers": [
                    {
                      "text": "A myth is that as a woman you have to be covered completely and wear hijab. Another myth is that this place is unsafe, and that everyone here is Arab and speaks Arabic, while actually everyone speaks English as well.",
                      "answer_start": 85
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are the craziest myths that you've heard before coming here about this country? A myth is that as a woman you have to be covered completely and wear hijab. Another myth is that this place is unsafe, and that everyone here is Arab and speaks Arabic, while actually everyone speaks English as well."
            },
            {
              "qas": [
                {
                  "question": "What is the luckiest thing that has happened to you?",
                  "id": "id193",
                  "answers": [
                    {
                      "text": "Getting into NYUAD. Even if I think I fit in well here, I understand that a lot of amazing people applied here so I must have had a lot of luck.",
                      "answer_start": 53
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the luckiest thing that has happened to you? Getting into NYUAD. Even if I think I fit in well here, I understand that a lot of amazing people applied here so I must have had a lot of luck."
            },
            {
              "qas": [
                {
                  "question": "How did you convince your parents to study at NYUAD?",
                  "id": "id194",
                  "answers": [
                    {
                      "text": "I convinced my parents by talking to the friend that went to the university and he confirmed that it's the best place and that it's very safe.",
                      "answer_start": 53
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How did you convince your parents to study at NYUAD? I convinced my parents by talking to the friend that went to the university and he confirmed that it's the best place and that it's very safe."
            },
            {
              "qas": [
                {
                  "question": "What would you look like had you not come to NYUAD?",
                  "id": "id195",
                  "answers": [
                    {
                      "text": "I don't know. I don't want to know. I love this place and I love how I've grown.",
                      "answer_start": 52
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What would you look like had you not come to NYUAD? I don't know. I don't want to know. I love this place and I love how I've grown."
            },
            {
              "qas": [
                {
                  "question": "How do you feel like your perspective or like your values or like you as a person has changed since you first came here?",
                  "id": "id196",
                  "answers": [
                    {
                      "text": "I feel like I've definitely become more tolerant and accepting.",
                      "answer_start": 121
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do you feel like your perspective or like your values or like you as a person has changed since you first came here? I feel like I've definitely become more tolerant and accepting."
            },
            {
              "qas": [
                {
                  "question": "Have you infused programming into new projects throughout since you studied music technology and composition?",
                  "id": "id197",
                  "answers": [
                    {
                      "text": "I have used programming in one of my business classes in which I had to negotiate deals and needed to manipulate big numbers on the spot and I did not have time for that. So I created my own calculator through Python.",
                      "answer_start": 110
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you infused programming into new projects throughout since you studied music technology and composition? I have used programming in one of my business classes in which I had to negotiate deals and needed to manipulate big numbers on the spot and I did not have time for that. So I created my own calculator through Python."
            },
            {
              "qas": [
                {
                  "question": "Have you been in competitions?",
                  "id": "id198",
                  "answers": [
                    {
                      "text": "I have! My most notable ones were winning at an a cappella contest with my friends, and also taking first place at the national Ecology Olympiad in Moldova as a senior in high school.",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you been in competitions? I have! My most notable ones were winning at an a cappella contest with my friends, and also taking first place at the national Ecology Olympiad in Moldova as a senior in high school."
            },
            {
              "qas": [
                {
                  "question": "Have you made friends at the university?",
                  "id": "id199",
                  "answers": [
                    {
                      "text": "I sure made a lot of friends at university.",
                      "answer_start": 41
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you made friends at the university? I sure made a lot of friends at university."
            },
            {
              "qas": [
                {
                  "question": "So have you had any programming courses?",
                  "id": "id200",
                  "answers": [
                    {
                      "text": "I took one Intro to Computer Science class in which I studied the basics of Python.",
                      "answer_start": 41
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "So have you had any programming courses? I took one Intro to Computer Science class in which I studied the basics of Python."
            },
            {
              "qas": [
                {
                  "question": "Do you play volleyball for the team at NYUAD?",
                  "id": "id201",
                  "answers": [
                    {
                      "text": "I used to play volleyball at NYUAD, but I had to leave because I've injured my back because I wasn't serving correctly.",
                      "answer_start": 46
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you play volleyball for the team at NYUAD? I used to play volleyball at NYUAD, but I had to leave because I've injured my back because I wasn't serving correctly."
            },
            {
              "qas": [
                {
                  "question": "Introduce me your university. Like its name. Where have you studied.",
                  "id": "id202",
                  "answers": [
                    {
                      "text": "I went to New York University in Abu Dhabi. It's a liberal arts university. And there's actually three campuses: one in New York, one in Abu Dhabi, and one in Shanghai.",
                      "answer_start": 69
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Introduce me your university. Like its name. Where have you studied. I went to New York University in Abu Dhabi. It's a liberal arts university. And there's actually three campuses: one in New York, one in Abu Dhabi, and one in Shanghai."
            },
            {
              "qas": [
                {
                  "question": "Where would you have applied if not for NYUAD?",
                  "id": "id203",
                  "answers": [
                    {
                      "text": "I'd probably try some of the big universities like Harvard and Oxford, but also some lesser known universities around Europe.",
                      "answer_start": 47
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where would you have applied if not for NYUAD? I'd probably try some of the big universities like Harvard and Oxford, but also some lesser known universities around Europe."
            },
            {
              "qas": [
                {
                  "question": "What would you miss most about NYUAD?",
                  "id": "id204",
                  "answers": [
                    {
                      "text": "I'll miss my friends a lot and I'll miss constantly being in an environment with other intelligent, tolerant and like-minded people.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What would you miss most about NYUAD? I'll miss my friends a lot and I'll miss constantly being in an environment with other intelligent, tolerant and like-minded people."
            },
            {
              "qas": [
                {
                  "question": "What would you have studied if not music at NYUAD?",
                  "id": "id205",
                  "answers": [
                    {
                      "text": "If we had a program for interpreting, I would probably study languages.",
                      "answer_start": 51
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What would you have studied if not music at NYUAD? If we had a program for interpreting, I would probably study languages."
            },
            {
              "qas": [
                {
                  "question": "What's the hardest major at NYUAD?",
                  "id": "id206",
                  "answers": [
                    {
                      "text": "Maybe engineering, because students have to take a lot of science classes that don't directly relate to their major, but anything can be hard if you have high standards for your work.",
                      "answer_start": 35
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's the hardest major at NYUAD? Maybe engineering, because students have to take a lot of science classes that don't directly relate to their major, but anything can be hard if you have high standards for your work."
            },
            {
              "qas": [
                {
                  "question": "What's been the most memorable thing from being just here?",
                  "id": "id207",
                  "answers": [
                    {
                      "text": "Maybe it's because it's been so recent, but the graduation was spectacular and my family was here and saying goodbye to friends and all that. I've also loved of course all the study abroad experiences, they have changed me so much.",
                      "answer_start": 59
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's been the most memorable thing from being just here? Maybe it's because it's been so recent, but the graduation was spectacular and my family was here and saying goodbye to friends and all that. I've also loved of course all the study abroad experiences, they have changed me so much."
            },
            {
              "qas": [
                {
                  "question": "Did you consider going to a Moldovan university?",
                  "id": "id208",
                  "answers": [
                    {
                      "text": "No, I always had a burning desire to explore the world and find myself somewhere else.",
                      "answer_start": 49
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you consider going to a Moldovan university? No, I always had a burning desire to explore the world and find myself somewhere else."
            },
            {
              "qas": [
                {
                  "question": "What kind of university is NYUAD?",
                  "id": "id209",
                  "answers": [
                    {
                      "text": "NYUAD is a liberal arts university.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What kind of university is NYUAD? NYUAD is a liberal arts university."
            },
            {
              "qas": [
                {
                  "question": "So how long have you been in the UAE?",
                  "id": "id210",
                  "answers": [
                    {
                      "text": "So I came here four years ago, but because of all the traveling probably two years in total.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "So how long have you been in the UAE? So I came here four years ago, but because of all the traveling probably two years in total."
            },
            {
              "qas": [
                {
                  "question": "Will you miss studying?",
                  "id": "id211",
                  "answers": [
                    {
                      "text": "Yeah. It's crazy to say it, but I like not having to worry about food then getting to travel and just enriching my knowledge.",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Will you miss studying? Yeah. It's crazy to say it, but I like not having to worry about food then getting to travel and just enriching my knowledge."
            },
            {
              "qas": [
                {
                  "question": "Did the campus use to be in the city?",
                  "id": "id212",
                  "answers": [
                    {
                      "text": "Yes, it was! But the university needed to expand to accomodate the rising number of students and needed more facilities",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did the campus use to be in the city? Yes, it was! But the university needed to expand to accomodate the rising number of students and needed more facilities"
            },
            {
              "qas": [
                {
                  "question": "Is it hard to get into the university?",
                  "id": "id213",
                  "answers": [
                    {
                      "text": "Yes, the acceptance rate is low, so aside from being great you need a little bit of luck.",
                      "answer_start": 39
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is it hard to get into the university? Yes, the acceptance rate is low, so aside from being great you need a little bit of luck."
            },
            {
              "qas": [
                {
                  "question": "What movie title best describes your life?",
                  "id": "id214",
                  "answers": [
                    {
                      "text": "\"Wonder Woman\" because I always wonder what I should do next with my life.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What movie title best describes your life? \"Wonder Woman\" because I always wonder what I should do next with my life."
            },
            {
              "qas": [
                {
                  "question": "What kind of wine do you like?",
                  "id": "id215",
                  "answers": [
                    {
                      "text": "Any wine, really. The sweeter the better. The redder the better, too.",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What kind of wine do you like? Any wine, really. The sweeter the better. The redder the better, too."
            },
            {
              "qas": [
                {
                  "question": "You think that was worth it?",
                  "id": "id216",
                  "answers": [
                    {
                      "text": "Definitely! Now I have the academic knowledge AND real-world experience!",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "You think that was worth it? Definitely! Now I have the academic knowledge AND real-world experience!"
            },
            {
              "qas": [
                {
                  "question": "I'm assuming that you think that expertise and knowledge is more important than what there is on your transcript as if like major or minor. As long as you have the expertise that there's no real need for that major thing to be on your transcript?",
                  "id": "id217",
                  "answers": [
                    {
                      "text": "Exactly. The expertise is more important long-term, but your major can affect your entry-level options.",
                      "answer_start": 247
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "I'm assuming that you think that expertise and knowledge is more important than what there is on your transcript as if like major or minor. As long as you have the expertise that there's no real need for that major thing to be on your transcript? Exactly. The expertise is more important long-term, but your major can affect your entry-level options."
            },
            {
              "qas": [
                {
                  "question": "Was Music Theory hard?",
                  "id": "id218",
                  "answers": [
                    {
                      "text": "For me, no. But neither would it be for any passionate beginner!",
                      "answer_start": 23
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Was Music Theory hard? For me, no. But neither would it be for any passionate beginner!"
            },
            {
              "qas": [
                {
                  "question": "What are you most thankful for?",
                  "id": "id219",
                  "answers": [
                    {
                      "text": "I am thankful to have an amazing family, for having the opportunities that I have now to grow and for how lucky I've been in general so far.",
                      "answer_start": 32
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are you most thankful for? I am thankful to have an amazing family, for having the opportunities that I have now to grow and for how lucky I've been in general so far."
            },
            {
              "qas": [
                {
                  "question": "What inspires your art?",
                  "id": "id220",
                  "answers": [
                    {
                      "text": "I can find inspiration for art in anything: nature, emotions, procrastination or deadlines.",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What inspires your art? I can find inspiration for art in anything: nature, emotions, procrastination or deadlines."
            },
            {
              "qas": [
                {
                  "question": "Do you judge a book by its cover?",
                  "id": "id221",
                  "answers": [
                    {
                      "text": "I do my best not to, but of course I do in the beginning of meeting someone or learning a concept.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you judge a book by its cover? I do my best not to, but of course I do in the beginning of meeting someone or learning a concept."
            },
            {
              "qas": [
                {
                  "question": "Who is your idol?",
                  "id": "id222",
                  "answers": [
                    {
                      "text": "I don't have one idol, so I don't know how to reply to that.",
                      "answer_start": 18
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Who is your idol? I don't have one idol, so I don't know how to reply to that."
            },
            {
              "qas": [
                {
                  "question": "What is your biggest regret?",
                  "id": "id223",
                  "answers": [
                    {
                      "text": "I don't really regret anything. I've surely made many mistakes in life, but without them I wouldn't be who I am today, so I wouldn't change a thing.",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your biggest regret? I don't really regret anything. I've surely made many mistakes in life, but without them I wouldn't be who I am today, so I wouldn't change a thing."
            },
            {
              "qas": [
                {
                  "question": "OK. Do you think there is such a thing as giving back to your country or do you feel like a responsibility to go back and give back to your country? Personally?",
                  "id": "id224",
                  "answers": [
                    {
                      "text": "I personally feel that responsibility, but I don't think people owe anyone anything just because they were born in a certain spot.",
                      "answer_start": 161
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "OK. Do you think there is such a thing as giving back to your country or do you feel like a responsibility to go back and give back to your country? Personally? I personally feel that responsibility, but I don't think people owe anyone anything just because they were born in a certain spot."
            },
            {
              "qas": [
                {
                  "question": "What would you change about yourself if you could?",
                  "id": "id225",
                  "answers": [
                    {
                      "text": "I would become more patient and resilient. Although I feel like I already have those traits, I want to develop them more.",
                      "answer_start": 51
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What would you change about yourself if you could? I would become more patient and resilient. Although I feel like I already have those traits, I want to develop them more."
            },
            {
              "qas": [
                {
                  "question": "If you repeated your four years of NYUAD what would you change?",
                  "id": "id226",
                  "answers": [
                    {
                      "text": "I would have taken a computer science class earlier. I took it in my junior year and that was quite late to realize that I like it.",
                      "answer_start": 64
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "If you repeated your four years of NYUAD what would you change? I would have taken a computer science class earlier. I took it in my junior year and that was quite late to realize that I like it."
            },
            {
              "qas": [
                {
                  "question": "What would you do if you won the lottery?",
                  "id": "id227",
                  "answers": [
                    {
                      "text": "I would invest the money, become self-sustainable and then help the world as much as I can.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What would you do if you won the lottery? I would invest the money, become self-sustainable and then help the world as much as I can."
            },
            {
              "qas": [
                {
                  "question": "What are you most likely very wrong about?",
                  "id": "id228",
                  "answers": [
                    {
                      "text": "I'm probably wrong about the origins of the world.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are you most likely very wrong about? I'm probably wrong about the origins of the world."
            },
            {
              "qas": [
                {
                  "question": "What is your definition of successful?",
                  "id": "id229",
                  "answers": [
                    {
                      "text": "Improving myself and the relationships with other people is my definition of success.",
                      "answer_start": 39
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your definition of successful? Improving myself and the relationships with other people is my definition of success."
            },
            {
              "qas": [
                {
                  "question": "What's your proudest accomplishment?",
                  "id": "id230",
                  "answers": [
                    {
                      "text": "My biggest accomplishment is that I have learned how to be true to myself and not give in to peer pressure in any aspects of my life.",
                      "answer_start": 37
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your proudest accomplishment? My biggest accomplishment is that I have learned how to be true to myself and not give in to peer pressure in any aspects of my life."
            },
            {
              "qas": [
                {
                  "question": "What's your biggest fear?",
                  "id": "id231",
                  "answers": [
                    {
                      "text": "My biggest fear is going through life without meaning.",
                      "answer_start": 26
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your biggest fear? My biggest fear is going through life without meaning."
            },
            {
              "qas": [
                {
                  "question": "Is Boxing difficult?",
                  "id": "id232",
                  "answers": [
                    {
                      "text": "No, the class is completely doable.",
                      "answer_start": 21
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is Boxing difficult? No, the class is completely doable."
            },
            {
              "qas": [
                {
                  "question": "Who knows you the best?",
                  "id": "id233",
                  "answers": [
                    {
                      "text": "Probably my best friends.",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Who knows you the best? Probably my best friends."
            },
            {
              "qas": [
                {
                  "question": "What makes you laugh?",
                  "id": "id234",
                  "answers": [
                    {
                      "text": "Silly jokes and extremely well-crafted jokes.",
                      "answer_start": 22
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What makes you laugh? Silly jokes and extremely well-crafted jokes."
            },
            {
              "qas": [
                {
                  "question": "Did your parents like it here?",
                  "id": "id235",
                  "answers": [
                    {
                      "text": "They loved it! They looked so out of place and were making interesting comments. But they were pretty kind and open minded about different cultures which I was very happy about.",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did your parents like it here? They loved it! They looked so out of place and were making interesting comments. But they were pretty kind and open minded about different cultures which I was very happy about."
            },
            {
              "qas": [
                {
                  "question": "If your life was a book, what would its name",
                  "id": "id236",
                  "answers": [
                    {
                      "text": "This one's so silly... But \"The Peak\" because my name is the peak of a mountain in Uganda and I always try to reach the peak of my potential.",
                      "answer_start": 45
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "If your life was a book, what would its name This one's so silly... But \"The Peak\" because my name is the peak of a mountain in Uganda and I always try to reach the peak of my potential."
            },
            {
              "qas": [
                {
                  "question": "What really makes you angry?",
                  "id": "id237",
                  "answers": [
                    {
                      "text": "This question.",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What really makes you angry? This question."
            },
            {
              "qas": [
                {
                  "question": "So you came here four years ago of as a freshman. Do you think... Do you think you accomplished or got to where you wanted to be or the goals you set before you came here?",
                  "id": "id238",
                  "answers": [
                    {
                      "text": "Yes. Actually I think so. In freshman year we were all made to write a letter to our senior selves and I was shocked that I did even more than I expected of myself.",
                      "answer_start": 172
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "So you came here four years ago of as a freshman. Do you think... Do you think you accomplished or got to where you wanted to be or the goals you set before you came here? Yes. Actually I think so. In freshman year we were all made to write a letter to our senior selves and I was shocked that I did even more than I expected of myself."
            },
            {
              "qas": [
                {
                  "question": "What do you do when you have no inspiration?",
                  "id": "id239",
                  "answers": [
                    {
                      "text": "You work every day. Even if for only 20 minutes. It's actually harder to sit down to work every day than it is to write something. So make it a habit.",
                      "answer_start": 45
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What do you do when you have no inspiration? You work every day. Even if for only 20 minutes. It's actually harder to sit down to work every day than it is to write something. So make it a habit."
            },
            {
              "qas": [
                {
                  "question": "Was Music Technology hard?",
                  "id": "id240",
                  "answers": [
                    {
                      "text": "For me, no. But neither would it be for a any passionate beginner!",
                      "answer_start": 27
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Was Music Technology hard? For me, no. But neither would it be for a any passionate beginner!"
            },
            {
              "qas": [
                {
                  "question": "What's your biggest accomplishment?",
                  "id": "id241",
                  "answers": [
                    {
                      "text": "My biggest accomplishment is that I have learned how to be true to myself and not give in to peer pressure on any aspects of my life.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your biggest accomplishment? My biggest accomplishment is that I have learned how to be true to myself and not give in to peer pressure on any aspects of my life."
            },
            {
              "qas": [
                {
                  "question": "Do you like the food here?",
                  "id": "id242",
                  "answers": [
                    {
                      "text": "Sometimes it is little spicy for me. But in general if I tone it down a bit it's very delicious.",
                      "answer_start": 27
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you like the food here? Sometimes it is little spicy for me. But in general if I tone it down a bit it's very delicious."
            },
            {
              "qas": [
                {
                  "question": "How is the workload at NYU Abu Dhabi?",
                  "id": "id243",
                  "answers": [
                    {
                      "text": "I think the Abu Dhabi and the Shanghai campus are more intense in terms of studying compared to the New York campus.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How is the workload at NYU Abu Dhabi? I think the Abu Dhabi and the Shanghai campus are more intense in terms of studying compared to the New York campus."
            },
            {
              "qas": [
                {
                  "question": "How did NYU Abu Dhabi come about?",
                  "id": "id244",
                  "answers": [
                    {
                      "text": "In 2005, representatives of NYU and the Emirate of Abu Dhabi recognized that through a new institution, NYU Abu Dhabi, they could establish a truly global network university in the UAE.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How did NYU Abu Dhabi come about? In 2005, representatives of NYU and the Emirate of Abu Dhabi recognized that through a new institution, NYU Abu Dhabi, they could establish a truly global network university in the UAE."
            },
            {
              "qas": [
                {
                  "question": "Did you go to a special school?",
                  "id": "id245",
                  "answers": [
                    {
                      "text": "I did go to a francophone school, but it still was in my little village in my little country.",
                      "answer_start": 32
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did you go to a special school? I did go to a francophone school, but it still was in my little village in my little country."
            },
            {
              "qas": [
                {
                  "question": "What is the proudest accomplishment of your childhood?",
                  "id": "id246",
                  "answers": [
                    {
                      "text": "I started to read way before everyone else in my kindergarten grade.",
                      "answer_start": 55
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the proudest accomplishment of your childhood? I started to read way before everyone else in my kindergarten grade."
            },
            {
              "qas": [
                {
                  "question": "What was your favorite subject in school?",
                  "id": "id247",
                  "answers": [
                    {
                      "text": "I used to love maths, French and physical education.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was your favorite subject in school? I used to love maths, French and physical education."
            },
            {
              "qas": [
                {
                  "question": "What was your childhood dream?",
                  "id": "id248",
                  "answers": [
                    {
                      "text": "I wanted to be a pop singer and I was super sure that that's what I was going to become. I guess close enough!",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was your childhood dream? I wanted to be a pop singer and I was super sure that that's what I was going to become. I guess close enough!"
            },
            {
              "qas": [
                {
                  "question": "What was your first job?",
                  "id": "id249",
                  "answers": [
                    {
                      "text": "I was working at my dad's office, making photocopies when I was like 13.",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was your first job? I was working at my dad's office, making photocopies when I was like 13."
            },
            {
              "qas": [
                {
                  "question": "What is one thing you're glad you tried but would never do again?",
                  "id": "id250",
                  "answers": [
                    {
                      "text": "I worked as a coordinator for kids at an American summer camp in Japan for a month and a half in summer 2017, and it was great but super tiring.",
                      "answer_start": 66
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is one thing you're glad you tried but would never do again? I worked as a coordinator for kids at an American summer camp in Japan for a month and a half in summer 2017, and it was great but super tiring."
            },
            {
              "qas": [
                {
                  "question": "When have you felt your biggest adrenaline rush?",
                  "id": "id251",
                  "answers": [
                    {
                      "text": "When going down a waterslide the size of 14-story building.",
                      "answer_start": 49
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "When have you felt your biggest adrenaline rush? When going down a waterslide the size of 14-story building."
            },
            {
              "qas": [
                {
                  "question": "Tell me a story from your childhood!",
                  "id": "id252",
                  "answers": [
                    {
                      "text": "When I was about four, I saw someone playing the piano and I really wanted my parents to buy me one. My parents thought it's whim and pianos were expensive, so they refused. I didn't know how money worked, so I started collecting coins. One day I showed my mom that I had a fistfull of coins for her to buy me a piano and that was the day my parents decided that they really needed to buy me a piano.",
                      "answer_start": 37
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Tell me a story from your childhood! When I was about four, I saw someone playing the piano and I really wanted my parents to buy me one. My parents thought it's whim and pianos were expensive, so they refused. I didn't know how money worked, so I started collecting coins. One day I showed my mom that I had a fistfull of coins for her to buy me a piano and that was the day my parents decided that they really needed to buy me a piano."
            },
            {
              "qas": [
                {
                  "question": "What is your favorite childhood memory?",
                  "id": "id253",
                  "answers": [
                    {
                      "text": "When I was about four, I saw someone playing the piano and I really wanted my parents to buy me one. My parents refused, thinking that it's a whim and telling me that pianos are expensive, so me having no concept of what money means I started collecting coins. So one day I showed my mom that I had a fistfull of coins for her to buy me a piano! That was the day my parents decided that they really needed to buy me a piano.",
                      "answer_start": 40
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your favorite childhood memory? When I was about four, I saw someone playing the piano and I really wanted my parents to buy me one. My parents refused, thinking that it's a whim and telling me that pianos are expensive, so me having no concept of what money means I started collecting coins. So one day I showed my mom that I had a fistfull of coins for her to buy me a piano! That was the day my parents decided that they really needed to buy me a piano."
            },
            {
              "qas": [
                {
                  "question": "What was your favorite class in school?",
                  "id": "id254",
                  "answers": [
                    {
                      "text": "I used to love maths, French and Physical education.",
                      "answer_start": 40
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What was your favorite class in school? I used to love maths, French and Physical education."
            },
            {
              "qas": [
                {
                  "question": "Good day!",
                  "id": "id255",
                  "answers": [
                    {
                      "text": "Good day!",
                      "answer_start": 10
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Good day! Good day!"
            },
            {
              "qas": [
                {
                  "question": "Good evening!",
                  "id": "id256",
                  "answers": [
                    {
                      "text": "Good evening!",
                      "answer_start": 14
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Good evening! Good evening!"
            },
            {
              "qas": [
                {
                  "question": "Good morning!",
                  "id": "id257",
                  "answers": [
                    {
                      "text": "Good morning!",
                      "answer_start": 14
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Good morning! Good morning!"
            },
            {
              "qas": [
                {
                  "question": "Inshallah.",
                  "id": "id258",
                  "answers": [
                    {
                      "text": "Inshallah.",
                      "answer_start": 11
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Inshallah. Inshallah."
            },
            {
              "qas": [
                {
                  "question": "It was nice meeting you.",
                  "id": "id259",
                  "answers": [
                    {
                      "text": "Likewise! Thank you for talking to me!",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "It was nice meeting you. Likewise! Thank you for talking to me!"
            },
            {
              "qas": [
                {
                  "question": "Nice to meet you.",
                  "id": "id260",
                  "answers": [
                    {
                      "text": "Nice to meet you too!",
                      "answer_start": 18
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Nice to meet you. Nice to meet you too!"
            },
            {
              "qas": [
                {
                  "question": "Good to see you again.",
                  "id": "id261",
                  "answers": [
                    {
                      "text": "Nice to see you, too!",
                      "answer_start": 23
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Good to see you again. Nice to see you, too!"
            },
            {
              "qas": [
                {
                  "question": "Sorry.",
                  "id": "id262",
                  "answers": [
                    {
                      "text": "No no, it's okay!",
                      "answer_start": 7
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Sorry. No no, it's okay!"
            },
            {
              "qas": [
                {
                  "question": "Hmmm...",
                  "id": "id263",
                  "answers": [
                    {
                      "text": "No Worries. Take your time.",
                      "answer_start": 8
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Hmmm... No Worries. Take your time."
            },
            {
              "qas": [
                {
                  "question": "How are things?",
                  "id": "id264",
                  "answers": [
                    {
                      "text": "Not too bad, thanks.",
                      "answer_start": 16
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How are things? Not too bad, thanks."
            },
            {
              "qas": [
                {
                  "question": "What's up?",
                  "id": "id265",
                  "answers": [
                    {
                      "text": "Nothing much. Hope you're doing well.",
                      "answer_start": 11
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's up? Nothing much. Hope you're doing well."
            },
            {
              "qas": [
                {
                  "question": "How have you been?",
                  "id": "id266",
                  "answers": [
                    {
                      "text": "Pretty good, thank you!",
                      "answer_start": 19
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How have you been? Pretty good, thank you!"
            },
            {
              "qas": [
                {
                  "question": "Goodbye!",
                  "id": "id267",
                  "answers": [
                    {
                      "text": "See you later!",
                      "answer_start": 9
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Goodbye! See you later!"
            },
            {
              "qas": [
                {
                  "question": "Oh! Congratulations. That's so nice.",
                  "id": "id268",
                  "answers": [
                    {
                      "text": "Thank you so much.",
                      "answer_start": 37
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Oh! Congratulations. That's so nice. Thank you so much."
            },
            {
              "qas": [
                {
                  "question": "It's good to hear. Sounds very very good.",
                  "id": "id269",
                  "answers": [
                    {
                      "text": "Thank you.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "It's good to hear. Sounds very very good. Thank you."
            },
            {
              "qas": [
                {
                  "question": "That's nice. Very nice.",
                  "id": "id270",
                  "answers": [
                    {
                      "text": "Thanks!",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "That's nice. Very nice. Thanks!"
            },
            {
              "qas": [
                {
                  "question": "Enjoy your life, hey.",
                  "id": "id271",
                  "answers": [
                    {
                      "text": "Yep.",
                      "answer_start": 22
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Enjoy your life, hey. Yep."
            },
            {
              "qas": [
                {
                  "question": "My name is *",
                  "id": "id272",
                  "answers": [
                    {
                      "text": "Nice to meet you!",
                      "answer_start": 13
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "My name is * Nice to meet you!"
            },
            {
              "qas": [
                {
                  "question": "That's awesome.",
                  "id": "id273",
                  "answers": [
                    {
                      "text": "Thanks.",
                      "answer_start": 16
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "That's awesome. Thanks."
            },
            {
              "qas": [
                {
                  "question": "What is a capstone?",
                  "id": "id274",
                  "answers": [
                    {
                      "text": "A capstone is a year long research project that students are required to fulfill in their academic field in their last year. I am, in fact, the result of a capstone project.",
                      "answer_start": 20
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is a capstone? A capstone is a year long research project that students are required to fulfill in their academic field in their last year. I am, in fact, the result of a capstone project."
            },
            {
              "qas": [
                {
                  "question": "Is NYUAD co-educational?",
                  "id": "id275",
                  "answers": [
                    {
                      "text": "Classrooms at NYU Abu Dhabi are co-educational.",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is NYUAD co-educational? Classrooms at NYU Abu Dhabi are co-educational."
            },
            {
              "qas": [
                {
                  "question": "What are the graduating requirements?",
                  "id": "id276",
                  "answers": [
                    {
                      "text": "Graduating requirements differ from major to major so it depends on your situation.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are the graduating requirements? Graduating requirements differ from major to major so it depends on your situation."
            },
            {
              "qas": [
                {
                  "question": "Are NYUNY students able to study away in Abu Dhabi?",
                  "id": "id277",
                  "answers": [
                    {
                      "text": "NYU New York and Shanghai students can apply for a semester at NYU Abu Dhabi.",
                      "answer_start": 52
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are NYUNY students able to study away in Abu Dhabi? NYU New York and Shanghai students can apply for a semester at NYU Abu Dhabi."
            },
            {
              "qas": [
                {
                  "question": "Are NYUAD students able to study away at NYU global sites?",
                  "id": "id278",
                  "answers": [
                    {
                      "text": "NYUAD students can apply to study away at any of NYU's 14 global sites.",
                      "answer_start": 59
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are NYUAD students able to study away at NYU global sites? NYUAD students can apply to study away at any of NYU's 14 global sites."
            },
            {
              "qas": [
                {
                  "question": "Do the norms of academic freedom prevail at NYUAD?",
                  "id": "id279",
                  "answers": [
                    {
                      "text": "Of course they do.",
                      "answer_start": 51
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do the norms of academic freedom prevail at NYUAD? Of course they do."
            },
            {
              "qas": [
                {
                  "question": "Is there a curfew for people who live in dorms?",
                  "id": "id280",
                  "answers": [
                    {
                      "text": "Our dorms have no curfews!",
                      "answer_start": 48
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is there a curfew for people who live in dorms? Our dorms have no curfews!"
            },
            {
              "qas": [
                {
                  "question": "How did you find * ?",
                  "id": "id281",
                  "answers": [
                    {
                      "text": "I have nothing to complain about.",
                      "answer_start": 21
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How did you find * ? I have nothing to complain about."
            },
            {
              "qas": [
                {
                  "question": "Who is the President of NYU?",
                  "id": "id282",
                  "answers": [
                    {
                      "text": "As of January 2016, Andrew Hamilton is the president of New York University.",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Who is the President of NYU? As of January 2016, Andrew Hamilton is the president of New York University."
            },
            {
              "qas": [
                {
                  "question": "Who is John Sexton?",
                  "id": "id283",
                  "answers": [
                    {
                      "text": "John Sexton is the President Emeritus of New York University.",
                      "answer_start": 20
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Who is John Sexton? John Sexton is the President Emeritus of New York University."
            },
            {
              "qas": [
                {
                  "question": "Do you have a favorite professor?",
                  "id": "id284",
                  "answers": [
                    {
                      "text": "I have more favorite professors I guess. The professor from the negotiation course is pretty amazing, and so is my professor of strategic management.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have a favorite professor? I have more favorite professors I guess. The professor from the negotiation course is pretty amazing, and so is my professor of strategic management."
            },
            {
              "qas": [
                {
                  "question": "Where do NYUAD Professors come from?",
                  "id": "id285",
                  "answers": [
                    {
                      "text": "Just like the students, NYU Abu Dhabi professors come from all over the world.",
                      "answer_start": 37
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where do NYUAD Professors come from? Just like the students, NYU Abu Dhabi professors come from all over the world."
            },
            {
              "qas": [
                {
                  "question": "Are professors helpful?",
                  "id": "id286",
                  "answers": [
                    {
                      "text": "Professors here are amazing. I got all of my internships, research opportunities and even a full-time job with their help.",
                      "answer_start": 24
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are professors helpful? Professors here are amazing. I got all of my internships, research opportunities and even a full-time job with their help."
            },
            {
              "qas": [
                {
                  "question": "What's something you like to do the old-fashioned way?",
                  "id": "id287",
                  "answers": [
                    {
                      "text": "I like to squeeze orange juice by hand with the little plastic bowl-thingie, with no electricity.",
                      "answer_start": 55
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's something you like to do the old-fashioned way? I like to squeeze orange juice by hand with the little plastic bowl-thingie, with no electricity."
            },
            {
              "qas": [
                {
                  "question": "How long does it take you to get ready in the morning?",
                  "id": "id288",
                  "answers": [
                    {
                      "text": "I need roughly an hour, but I can also do it in 30 mins if I don't feel like looking good.",
                      "answer_start": 55
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How long does it take you to get ready in the morning? I need roughly an hour, but I can also do it in 30 mins if I don't feel like looking good."
            },
            {
              "qas": [
                {
                  "question": "Do you have any pets?",
                  "id": "id289",
                  "answers": [
                    {
                      "text": "I personally don't, but I really want pets in the future!",
                      "answer_start": 22
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have any pets? I personally don't, but I really want pets in the future!"
            },
            {
              "qas": [
                {
                  "question": "How often do you play sports?",
                  "id": "id290",
                  "answers": [
                    {
                      "text": "I play sports at least twice a week.",
                      "answer_start": 30
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How often do you play sports? I play sports at least twice a week."
            },
            {
              "qas": [
                {
                  "question": "What position in volleyball do you play?",
                  "id": "id291",
                  "answers": [
                    {
                      "text": "In school in Moldova we were taught to play all volleyball positions, so I don't have a particular favorite position.",
                      "answer_start": 41
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What position in volleyball do you play? In school in Moldova we were taught to play all volleyball positions, so I don't have a particular favorite position."
            },
            {
              "qas": [
                {
                  "question": "What year is it?",
                  "id": "id292",
                  "answers": [
                    {
                      "text": "It's 2019 here!",
                      "answer_start": 17
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What year is it? It's 2019 here!"
            },
            {
              "qas": [
                {
                  "question": "What is your spirit animal?",
                  "id": "id293",
                  "answers": [
                    {
                      "text": "Maybe a bird. I love traveling, being in the sky and feeling free.",
                      "answer_start": 28
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your spirit animal? Maybe a bird. I love traveling, being in the sky and feeling free."
            },
            {
              "qas": [
                {
                  "question": "What website do you visit most often?",
                  "id": "id294",
                  "answers": [
                    {
                      "text": "My most visited websites are facebook, gmail, and 9gag.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What website do you visit most often? My most visited websites are facebook, gmail, and 9gag."
            },
            {
              "qas": [
                {
                  "question": "What's your favorite holiday?",
                  "id": "id295",
                  "answers": [
                    {
                      "text": "New Year's Eve, which is when we celebrate with gifts, Christmas trees, Santa Claus, and drinks.",
                      "answer_start": 30
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's your favorite holiday? New Year's Eve, which is when we celebrate with gifts, Christmas trees, Santa Claus, and drinks."
            },
            {
              "qas": [
                {
                  "question": "Can you understand me?",
                  "id": "id296",
                  "answers": [
                    {
                      "text": "Of course I can understand you.",
                      "answer_start": 23
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Can you understand me? Of course I can understand you."
            },
            {
              "qas": [
                {
                  "question": "What is solyanka?",
                  "id": "id297",
                  "answers": [
                    {
                      "text": "Solyanka is a Russian soup with tomatoes, pickles, sausages and herbs. It's delicious!",
                      "answer_start": 18
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is solyanka? Solyanka is a Russian soup with tomatoes, pickles, sausages and herbs. It's delicious!"
            },
            {
              "qas": [
                {
                  "question": "Are you religious?",
                  "id": "id298",
                  "answers": [
                    {
                      "text": "I am sure something bigger than us exists, but I don't follow any specific book or rituals.",
                      "answer_start": 19
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are you religious? I am sure something bigger than us exists, but I don't follow any specific book or rituals."
            },
            {
              "qas": [
                {
                  "question": "What did you think of when you first heard about Ramadan?",
                  "id": "id299",
                  "answers": [
                    {
                      "text": "I didn't know much about it before, but it wasn't something shocking and I even joined some of my friends at the Mosque for Eid.",
                      "answer_start": 58
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What did you think of when you first heard about Ramadan? I didn't know much about it before, but it wasn't something shocking and I even joined some of my friends at the Mosque for Eid."
            },
            {
              "qas": [
                {
                  "question": "Did your religion change after coming here and being being exposed to like so many other types of religions?",
                  "id": "id300",
                  "answers": [
                    {
                      "text": "I started questioning my beliefs when I was a teenager after watching lots of documentaries and talking to agnostics alike.",
                      "answer_start": 109
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Did your religion change after coming here and being being exposed to like so many other types of religions? I started questioning my beliefs when I was a teenager after watching lots of documentaries and talking to agnostics alike."
            },
            {
              "qas": [
                {
                  "question": "Do you ever think that you might go back to being or would you practice any sort of religious rituals or things like that?",
                  "id": "id301",
                  "answers": [
                    {
                      "text": "Sure. I'm not against religions, but I'd have to have a strong reason to convert to one.",
                      "answer_start": 123
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you ever think that you might go back to being or would you practice any sort of religious rituals or things like that? Sure. I'm not against religions, but I'd have to have a strong reason to convert to one."
            },
            {
              "qas": [
                {
                  "question": "Are most people back home Orthodox?",
                  "id": "id302",
                  "answers": [
                    {
                      "text": "They're mostly Christian Orthodox, but there's also Protestants and Seventh Day Adventists and some versions of those.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are most people back home Orthodox? They're mostly Christian Orthodox, but there's also Protestants and Seventh Day Adventists and some versions of those."
            },
            {
              "qas": [
                {
                  "question": "How safe is it living in Abu Dhabi?",
                  "id": "id303",
                  "answers": [
                    {
                      "text": "Abu Dhabi is one of the safest cities in the world.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How safe is it living in Abu Dhabi? Abu Dhabi is one of the safest cities in the world."
            },
            {
              "qas": [
                {
                  "question": "How do you feel about being a woman in the UAE as a student?",
                  "id": "id304",
                  "answers": [
                    {
                      "text": "I feel safe, but women here do get a lot of unwanted stares.",
                      "answer_start": 61
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do you feel about being a woman in the UAE as a student? I feel safe, but women here do get a lot of unwanted stares."
            },
            {
              "qas": [
                {
                  "question": "When does this semester start?",
                  "id": "id305",
                  "answers": [
                    {
                      "text": "The academic year usually starts at the end of August.",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "When does this semester start? The academic year usually starts at the end of August."
            },
            {
              "qas": [
                {
                  "question": "Do you have to be here during the summer?",
                  "id": "id306",
                  "answers": [
                    {
                      "text": "You don't have to, but it surely looks good on the CV to do some internships or research.",
                      "answer_start": 42
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have to be here during the summer? You don't have to, but it surely looks good on the CV to do some internships or research."
            },
            {
              "qas": [
                {
                  "question": "It's very bold.",
                  "id": "id307",
                  "answers": [
                    {
                      "text": "(Silent nod)",
                      "answer_start": 16
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "It's very bold. (Silent nod)"
            },
            {
              "qas": [
                {
                  "question": "Exactly I could do anything.",
                  "id": "id308",
                  "answers": [
                    {
                      "text": "For sure!",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Exactly I could do anything. For sure!"
            },
            {
              "qas": [
                {
                  "question": "Yeah. But not for other people. Traditionally no ... People live their whole lives in the same place.",
                  "id": "id309",
                  "answers": [
                    {
                      "text": "Kind of.",
                      "answer_start": 102
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Yeah. But not for other people. Traditionally no ... People live their whole lives in the same place. Kind of."
            },
            {
              "qas": [
                {
                  "question": "Do they mark everything?",
                  "id": "id310",
                  "answers": [
                    {
                      "text": "Not really.",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do they mark everything? Not really."
            },
            {
              "qas": [
                {
                  "question": "My siblings?",
                  "id": "id311",
                  "answers": [
                    {
                      "text": "Sure.",
                      "answer_start": 13
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "My siblings? Sure."
            },
            {
              "qas": [
                {
                  "question": "How do people commute between campus and the city?",
                  "id": "id312",
                  "answers": [
                    {
                      "text": "Our students use NYUAD provided shuttles, taxis or the city bus.",
                      "answer_start": 51
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do people commute between campus and the city? Our students use NYUAD provided shuttles, taxis or the city bus."
            },
            {
              "qas": [
                {
                  "question": "Have you ever been to non-Western countries?",
                  "id": "id313",
                  "answers": [
                    {
                      "text": "I am from Moldova.",
                      "answer_start": 45
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you ever been to non-Western countries? I am from Moldova."
            },
            {
              "qas": [
                {
                  "question": "What did you do in New York?",
                  "id": "id314",
                  "answers": [
                    {
                      "text": "I had a marketing internship at a small music venue called DROM.",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What did you do in New York? I had a marketing internship at a small music venue called DROM."
            },
            {
              "qas": [
                {
                  "question": "Yes. So have you been to different places in the UAE?",
                  "id": "id315",
                  "answers": [
                    {
                      "text": "I have only been to Abu Dhabi, Dubai and Ras Al Khaimah.",
                      "answer_start": 54
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Yes. So have you been to different places in the UAE? I have only been to Abu Dhabi, Dubai and Ras Al Khaimah."
            },
            {
              "qas": [
                {
                  "question": "What did you like about Argentina?",
                  "id": "id316",
                  "answers": [
                    {
                      "text": "I loved the people, the culture and the fact that I've learned some Spanish!",
                      "answer_start": 35
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What did you like about Argentina? I loved the people, the culture and the fact that I've learned some Spanish!"
            },
            {
              "qas": [
                {
                  "question": "Have you ever been to like third world countries?",
                  "id": "id317",
                  "answers": [
                    {
                      "text": "I mean, that definition is subjective. My home country is a developing nation.",
                      "answer_start": 50
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Have you ever been to like third world countries? I mean, that definition is subjective. My home country is a developing nation."
            },
            {
              "qas": [
                {
                  "question": "What did you do in China?",
                  "id": "id318",
                  "answers": [
                    {
                      "text": "I went to China for my January term in January 2017 to study a class for about two weeks!",
                      "answer_start": 26
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What did you do in China? I went to China for my January term in January 2017 to study a class for about two weeks!"
            },
            {
              "qas": [
                {
                  "question": "Where have you been in Oman?",
                  "id": "id319",
                  "answers": [
                    {
                      "text": "I went to the Royal Opera House, Wadi Shab, and a few other places in Muscat. It was gorgeous!",
                      "answer_start": 29
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where have you been in Oman? I went to the Royal Opera House, Wadi Shab, and a few other places in Muscat. It was gorgeous!"
            },
            {
              "qas": [
                {
                  "question": "All those four years you haven't seen more Emirates?",
                  "id": "id320",
                  "answers": [
                    {
                      "text": "I wish I could see more places, but I'm not too crazy about traveling for brief periods of time. I prefer spending at least a month or half a year in a new place.",
                      "answer_start": 53
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "All those four years you haven't seen more Emirates? I wish I could see more places, but I'm not too crazy about traveling for brief periods of time. I prefer spending at least a month or half a year in a new place."
            },
            {
              "qas": [
                {
                  "question": "What about travelling?",
                  "id": "id321",
                  "answers": [
                    {
                      "text": "I've been to Buenos Aires and New York for a semester each and I've done two January terms, in Shanghai and Paris.",
                      "answer_start": 23
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What about travelling? I've been to Buenos Aires and New York for a semester each and I've done two January terms, in Shanghai and Paris."
            },
            {
              "qas": [
                {
                  "question": "What is the nightlife in Abu Dhabi like?",
                  "id": "id322",
                  "answers": [
                    {
                      "text": "It's quite peaceful, but I don't go out a lot at night.",
                      "answer_start": 41
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the nightlife in Abu Dhabi like? It's quite peaceful, but I don't go out a lot at night."
            },
            {
              "qas": [
                {
                  "question": "And which country that you've visited is your favorite one so far?",
                  "id": "id323",
                  "answers": [
                    {
                      "text": "My favorite country was Argentina!",
                      "answer_start": 67
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "And which country that you've visited is your favorite one so far? My favorite country was Argentina!"
            },
            {
              "qas": [
                {
                  "question": "Do you go to Dubai a lot?",
                  "id": "id324",
                  "answers": [
                    {
                      "text": "Sometimes! Some of my closest friends live there.",
                      "answer_start": 26
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you go to Dubai a lot? Sometimes! Some of my closest friends live there."
            },
            {
              "qas": [
                {
                  "question": "What is the perfect vacation place?",
                  "id": "id325",
                  "answers": [
                    {
                      "text": "The perfect vacation would be anywhere warm with beaches, greenery and waterfalls.",
                      "answer_start": 36
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the perfect vacation place? The perfect vacation would be anywhere warm with beaches, greenery and waterfalls."
            },
            {
              "qas": [
                {
                  "question": "Where have you travelled?",
                  "id": "id326",
                  "answers": [
                    {
                      "text": "I've been to Buenos Aires and New York for a semester each and I've done to January terms in Shanghai and Paris.",
                      "answer_start": 26
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Where have you travelled? I've been to Buenos Aires and New York for a semester each and I've done to January terms in Shanghai and Paris."
            },
            {
              "qas": [
                {
                  "question": "What are NYU Global Sites?",
                  "id": "id327",
                  "answers": [
                    {
                      "text": "NYUAD students can apply to study away at any of NYU's 14 global sites, the university's study away opportunities around the globe.",
                      "answer_start": 27
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What are NYU Global Sites? NYUAD students can apply to study away at any of NYU's 14 global sites, the university's study away opportunities around the globe."
            },
            {
              "qas": [
                {
                  "question": "Are you a clean or messy person?",
                  "id": "id328",
                  "answers": [
                    {
                      "text": "I am super clean, and I love to say that my room is the state of my mind - once something goes wrong, my room becomes messy.",
                      "answer_start": 33
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are you a clean or messy person? I am super clean, and I love to say that my room is the state of my mind - once something goes wrong, my room becomes messy."
            },
            {
              "qas": [
                {
                  "question": "What is your personality like?",
                  "id": "id329",
                  "answers": [
                    {
                      "text": "I am usually the clown in my friend groups so I think I'm extroverted, but I like to be alone as well.",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is your personality like? I am usually the clown in my friend groups so I think I'm extroverted, but I like to be alone as well."
            },
            {
              "qas": [
                {
                  "question": "Do you have a pet peeve?",
                  "id": "id330",
                  "answers": [
                    {
                      "text": "I don't like it when people interrupt.",
                      "answer_start": 25
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have a pet peeve? I don't like it when people interrupt."
            },
            {
              "qas": [
                {
                  "question": "Is it hard or easy to make you happy?",
                  "id": "id331",
                  "answers": [
                    {
                      "text": "I get excited at small things, so I guess it's easy.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Is it hard or easy to make you happy? I get excited at small things, so I guess it's easy."
            },
            {
              "qas": [
                {
                  "question": "Are you a romantic person?",
                  "id": "id332",
                  "answers": [
                    {
                      "text": "Probably yes. But what I can tell you for sure is that I am really affectionate.",
                      "answer_start": 27
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Are you a romantic person? Probably yes. But what I can tell you for sure is that I am really affectionate."
            },
            {
              "qas": [
                {
                  "question": "What's the tallest building you've been to the top in?",
                  "id": "id333",
                  "answers": [
                    {
                      "text": "Burj Khalifa! Obviously!",
                      "answer_start": 55
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What's the tallest building you've been to the top in? Burj Khalifa! Obviously!"
            },
            {
              "qas": [
                {
                  "question": "Do you have to wear a head-scarf",
                  "id": "id334",
                  "answers": [
                    {
                      "text": "Not at all! Abu Dhabi is very progressive and non-judgmental.",
                      "answer_start": 33
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you have to wear a head-scarf Not at all! Abu Dhabi is very progressive and non-judgmental."
            },
            {
              "qas": [
                {
                  "question": "Do you go out in Abu Dhabi a lot?",
                  "id": "id335",
                  "answers": [
                    {
                      "text": "Not too often, because I have a lot of work and I like the life on campus, but I do sometimes try to get away.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Do you go out in Abu Dhabi a lot? Not too often, because I have a lot of work and I like the life on campus, but I do sometimes try to get away."
            },
            {
              "qas": [
                {
                  "question": "What is the climate like in Abu Dhabi?",
                  "id": "id336",
                  "answers": [
                    {
                      "text": "The climate of Abu Dhabi is hot and arid. The temperatures range from 18 to 50 degrees Celsius.",
                      "answer_start": 39
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the climate like in Abu Dhabi? The climate of Abu Dhabi is hot and arid. The temperatures range from 18 to 50 degrees Celsius."
            },
            {
              "qas": [
                {
                  "question": "What is the currency used in the UAE?",
                  "id": "id337",
                  "answers": [
                    {
                      "text": "The currency of the UAE is the dirham. The UAE dirham is fixed to the US dollar at a rate of 3.67 dirham to the dollar.",
                      "answer_start": 38
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What is the currency used in the UAE? The currency of the UAE is the dirham. The UAE dirham is fixed to the US dollar at a rate of 3.67 dirham to the dollar."
            },
            {
              "qas": [
                {
                  "question": "Does the UAE relate itself to a specific religion?",
                  "id": "id338",
                  "answers": [
                    {
                      "text": "The official religion of the UAE is Islam, although other religions are respected and practiced freely.",
                      "answer_start": 51
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Does the UAE relate itself to a specific religion? The official religion of the UAE is Islam, although other religions are respected and practiced freely."
            },
            {
              "qas": [
                {
                  "question": "What type of government does the UAE have?",
                  "id": "id339",
                  "answers": [
                    {
                      "text": "The United Arab Emirates has a federal government with seven emirates. The head of the federal government is the president Sheikh Khalifa bin Zayed Al Nahyan.",
                      "answer_start": 43
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What type of government does the UAE have? The United Arab Emirates has a federal government with seven emirates. The head of the federal government is the president Sheikh Khalifa bin Zayed Al Nahyan."
            },
            {
              "qas": [
                {
                  "question": "Tell me a little about the history of the UAE.",
                  "id": "id340",
                  "answers": [
                    {
                      "text": "The United Arab Emirates was established as a nation in 1971 when the British withdrew from the region. The late Sheikh Zayed bin Sultan Al Nahyan, the founder and first president of the UAE, negotiated agreements between the emirates that now make up the nation.",
                      "answer_start": 47
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Tell me a little about the history of the UAE. The United Arab Emirates was established as a nation in 1971 when the British withdrew from the region. The late Sheikh Zayed bin Sultan Al Nahyan, the founder and first president of the UAE, negotiated agreements between the emirates that now make up the nation."
            },
            {
              "qas": [
                {
                  "question": "Can women drive in the UAE?",
                  "id": "id341",
                  "answers": [
                    {
                      "text": "Women can drive in the UAE.",
                      "answer_start": 28
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "Can women drive in the UAE? Women can drive in the UAE."
            },
            {
              "qas": [
                {
                  "question": "What can I do around Saadiyat?",
                  "id": "id342",
                  "answers": [
                    {
                      "text": "You can go to the beach, bars or visit the Louvre museum!",
                      "answer_start": 31
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What can I do around Saadiyat? You can go to the beach, bars or visit the Louvre museum!"
            },
            {
              "qas": [
                {
                  "question": "So what did you think feminism was before you came here?",
                  "id": "id343",
                  "answers": [
                    {
                      "text": "I don't know how to answer that.",
                      "answer_start": 57
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "So what did you think feminism was before you came here? I don't know how to answer that."
            },
            {
              "qas": [
                {
                  "question": "How do you measure?",
                  "id": "id344",
                  "answers": [
                    {
                      "text": "I don't really know.",
                      "answer_start": 20
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How do you measure? I don't really know."
            },
            {
              "qas": [
                {
                  "question": "I really like that ... one Song .. is it Romanian Folk Song? Is it The Bartok?",
                  "id": "id345",
                  "answers": [
                    {
                      "text": "I'm not sure.",
                      "answer_start": 79
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "I really like that ... one Song .. is it Romanian Folk Song? Is it The Bartok? I'm not sure."
            },
            {
              "qas": [
                {
                  "question": "How about the apartments and the offices being built in front?",
                  "id": "id346",
                  "answers": [
                    {
                      "text": "What do you mean?",
                      "answer_start": 63
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "How about the apartments and the offices being built in front? What do you mean?"
            },
            {
              "qas": [
                {
                  "question": "So what else can I ask you about?",
                  "id": "id347",
                  "answers": [
                    {
                      "text": "You can ask me about my family or about my passions. Anything you want.",
                      "answer_start": 34
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "So what else can I ask you about? You can ask me about my family or about my passions. Anything you want."
            },
            {
              "qas": [
                {
                  "question": "What can I talk to you about?",
                  "id": "id348",
                  "answers": [
                    {
                      "text": "You can ask me about the languages that I speak or my experiences in music.",
                      "answer_start": 30
                    }
                  ],
                  "is_impossible": false
                }
              ],
              "context": "What can I talk to you about? You can ask me about the languages that I speak or my experiences in music."
            }
          ]
        }
      ]
    }



```python
# Set documents such that the whole doc is Q + A

from typing import List
from haystack import Document

titles = df.text.to_list()
texts = ["{} {}".format(a, b) for a, b in zip(titles, df.answer.to_list())]
documents: List[Document] = []
for title, text in zip(titles, texts):
    documents.append(
        Document(
            text=text,
            meta={
                "name": title or ""
            }
        )
    )
```


```python
!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension
```

    Collecting ipywidgets
      Downloading ipywidgets-7.6.3-py2.py3-none-any.whl (121 kB)
    [K     |████████████████████████████████| 121 kB 6.7 MB/s eta 0:00:01
    [?25hRequirement already satisfied: nbformat>=4.2.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipywidgets) (5.0.8)
    Requirement already satisfied: ipykernel>=4.5.1 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipywidgets) (5.4.3)
    Requirement already satisfied: traitlets>=4.3.1 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipywidgets) (5.0.5)
    Requirement already satisfied: ipython>=4.0.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipywidgets) (7.19.0)
    Requirement already satisfied: tornado>=4.2 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1)
    Requirement already satisfied: jupyter-client in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1.11)
    Requirement already satisfied: appnope in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.2)
    Requirement already satisfied: pexpect>4.3 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipython>=4.0.0->ipywidgets) (4.8.0)
    Requirement already satisfied: pickleshare in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipython>=4.0.0->ipywidgets) (0.7.5)
    Requirement already satisfied: setuptools>=18.5 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipython>=4.0.0->ipywidgets) (51.1.2.post20210110)
    Requirement already satisfied: decorator in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipython>=4.0.0->ipywidgets) (4.4.2)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipython>=4.0.0->ipywidgets) (3.0.10)
    Requirement already satisfied: pygments in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipython>=4.0.0->ipywidgets) (2.7.4)
    Requirement already satisfied: jedi>=0.10 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipython>=4.0.0->ipywidgets) (0.18.0)
    Requirement already satisfied: backcall in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from ipython>=4.0.0->ipywidgets) (0.2.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from jedi>=0.10->ipython>=4.0.0->ipywidgets) (0.8.1)
    Collecting jupyterlab-widgets>=1.0.0
      Downloading jupyterlab_widgets-1.0.0-py3-none-any.whl (243 kB)
    [K     |████████████████████████████████| 243 kB 27.9 MB/s eta 0:00:01
    [?25hRequirement already satisfied: ipython-genutils in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets) (0.2.0)
    Requirement already satisfied: jupyter-core in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets) (4.7.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets) (3.2.0)
    Requirement already satisfied: pyrsistent>=0.14.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (0.17.3)
    Requirement already satisfied: importlib-metadata in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (3.4.0)
    Requirement already satisfied: six>=1.11.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (1.15.0)
    Requirement already satisfied: attrs>=17.4.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (20.3.0)
    Requirement already satisfied: ptyprocess>=0.5 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from pexpect>4.3->ipython>=4.0.0->ipywidgets) (0.7.0)
    Requirement already satisfied: wcwidth in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0->ipywidgets) (0.2.5)
    Collecting widgetsnbextension~=3.5.0
      Using cached widgetsnbextension-3.5.1-py2.py3-none-any.whl (2.2 MB)
    Requirement already satisfied: notebook>=4.4.1 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from widgetsnbextension~=3.5.0->ipywidgets) (6.1.6)
    Requirement already satisfied: argon2-cffi in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (20.1.0)
    Requirement already satisfied: nbconvert in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (6.0.7)
    Requirement already satisfied: Send2Trash in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.5.0)
    Requirement already satisfied: prometheus-client in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.9.0)
    Requirement already satisfied: jinja2 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.11.2)
    Requirement already satisfied: terminado>=0.8.3 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.9.2)
    Requirement already satisfied: pyzmq>=17 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (20.0.0)
    Requirement already satisfied: python-dateutil>=2.1 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from jupyter-client->ipykernel>=4.5.1->ipywidgets) (2.8.1)
    Requirement already satisfied: cffi>=1.0.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.14.4)
    Requirement already satisfied: pycparser in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from cffi>=1.0.0->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.20)
    Requirement already satisfied: typing-extensions>=3.6.4 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from importlib-metadata->jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (3.7.4.3)
    Requirement already satisfied: zipp>=0.5 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from importlib-metadata->jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (3.4.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from jinja2->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.1.1)
    Requirement already satisfied: pandocfilters>=1.4.1 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.4.3)
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.1)
    Requirement already satisfied: defusedxml in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.6.0)
    Requirement already satisfied: bleach in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (3.2.1)
    Requirement already satisfied: entrypoints>=0.2.2 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.3)
    Requirement already satisfied: jupyterlab-pygments in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.1.2)
    Requirement already satisfied: mistune<2,>=0.8.1 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.8.4)
    Requirement already satisfied: testpath in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.4.4)
    Requirement already satisfied: async-generator in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.10)
    Requirement already satisfied: nest-asyncio in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.4.3)
    Requirement already satisfied: webencodings in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.1)
    Requirement already satisfied: packaging in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (20.8)
    Requirement already satisfied: pyparsing>=2.0.2 in /Users/amc/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages (from packaging->bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.4.7)
    Installing collected packages: widgetsnbextension, jupyterlab-widgets, ipywidgets
    Successfully installed ipywidgets-7.6.3 jupyterlab-widgets-1.0.0 widgetsnbextension-3.5.1
    Enabling notebook extension jupyter-js-widgets/extension...
          - Validating: [32mOK[0m



```python
from haystack.generator.transformers import RAGenerator
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.transformers import TransformersReader


# qa_document_store = FAISSDocumentStore(
#     sql_url="postgresql://ironman:kolomino@localhost:5432/squadformat?client_encoding=utf8",
#     faiss_index_factory_str="Flat",
#     return_embedding=True
# )

# qa_retriever = DensePassageRetriever(
#     document_store=qa_document_store,
#     query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
#     passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
#     use_gpu=False,
#     embed_title=True,
# )

# qa_generator = RAGenerator(
#     model_name_or_path="facebook/rag-token-nq",
#     use_gpu=False,
#     top_k_answers=1,
#     max_length=200,
#     min_length=2,
#     embed_title=True,
#     num_beams=2,
# )

qa_reader = TransformersReader("deepset/roberta-base-squad2")

```

    02/01/2021 17:01:28 - INFO - filelock -   Lock 140337376916432 acquired on /Users/amc/.cache/torch/transformers/f7d4b9379a9c487fa03ccf3d8e00058faa9d664cf01fc03409138246f48760da.6060f348ba2b58d6d30b5324910152ffc512e7c3891ed13f22844f1a9b5c0d0f.lock
    02/01/2021 17:01:29 - INFO - filelock -   Lock 140337376916432 released on /Users/amc/.cache/torch/transformers/f7d4b9379a9c487fa03ccf3d8e00058faa9d664cf01fc03409138246f48760da.6060f348ba2b58d6d30b5324910152ffc512e7c3891ed13f22844f1a9b5c0d0f.lock



    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-208-7b51cd482f97> in <module>
         28 # )
         29 
    ---> 30 qa_reader = TransformersReader("deepset/roberta-base-squad2")
    

    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/haystack/reader/transformers.py in __init__(self, model_name_or_path, tokenizer, context_window_size, use_gpu, top_k_per_candidate, return_no_answers, max_seq_len, doc_stride)
         54 
         55         """
    ---> 56         self.model = pipeline('question-answering', model=model_name_or_path, tokenizer=tokenizer, device=use_gpu)
         57         self.context_window_size = context_window_size
         58         self.top_k_per_candidate = top_k_per_candidate


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/transformers/pipelines.py in pipeline(task, model, config, tokenizer, framework, **kwargs)
       2714             tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
       2715         else:
    -> 2716             tokenizer = AutoTokenizer.from_pretrained(tokenizer)
       2717 
       2718     # Instantiate config if needed


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/transformers/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
        214         config = kwargs.pop("config", None)
        215         if not isinstance(config, PretrainedConfig):
    --> 216             config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        217 
        218         if "bert-base-japanese" in str(pretrained_model_name_or_path):


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/transformers/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
        308             {'foo': False}
        309         """
    --> 310         config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        311 
        312         if "model_type" in config_dict:


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
        353                 proxies=proxies,
        354                 resume_download=resume_download,
    --> 355                 local_files_only=local_files_only,
        356             )
        357             # Load config dict


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/transformers/file_utils.py in cached_path(url_or_filename, cache_dir, force_download, proxies, resume_download, user_agent, extract_compressed_file, force_extract, local_files_only)
        721             resume_download=resume_download,
        722             user_agent=user_agent,
    --> 723             local_files_only=local_files_only,
        724         )
        725     elif os.path.exists(url_or_filename):


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/transformers/file_utils.py in get_from_cache(url, cache_dir, force_download, proxies, etag_timeout, resume_download, user_agent, local_files_only)
        900             logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)
        901 
    --> 902             http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)
        903 
        904         logger.info("storing %s in cache at %s", url, cache_path)


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/transformers/file_utils.py in http_get(url, temp_file, proxies, resume_size, user_agent)
        791         initial=resume_size,
        792         desc="Downloading",
    --> 793         disable=bool(logging.get_verbosity() == logging.NOTSET),
        794     )
        795     for chunk in response.iter_content(chunk_size=1024):


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/tqdm/notebook.py in __init__(self, *args, **kwargs)
        238         unit_scale = 1 if self.unit_scale is True else self.unit_scale or 1
        239         total = self.total * unit_scale if self.total else self.total
    --> 240         self.container = self.status_printer(self.fp, total, self.desc, self.ncols)
        241         self.container.pbar = self
        242         if display_here:


    ~/opt/miniconda3/envs/dm_api/lib/python3.7/site-packages/tqdm/notebook.py in status_printer(_, total, desc, ncols)
        116         if IProgress is None:  # #187 #451 #558 #872
        117             raise ImportError(
    --> 118                 "IProgress not found. Please update jupyter and ipywidgets."
        119                 " See https://ipywidgets.readthedocs.io/en/stable"
        120                 "/user_install.html")


    ImportError: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html



```python
qa_document_store.delete_all_documents()

qa_document_store.write_documents(documents)

qa_document_store.update_embeddings(
    retriever=qa_retriever
)
```

    01/31/2021 17:35:38 - INFO - haystack.document_store.faiss -   Updating embeddings for 349 docs...
    Creating Embeddings: 100%|██████████| 22/22 [01:23<00:00,  3.81s/ Batches]
    01/31/2021 17:37:02 - INFO - haystack.document_store.faiss -   Indexing embeddings and updating vectors_ids...
    100%|██████████| 1/1 [00:00<00:00,  7.73it/s]



```python
QUESTIONS = [q for q, hit in zip(test_questions, rr_test_hits_at_k) if hit == 0]
```


```python
# use finetune (I swapped the names)
# QUESTIONS = [q for q, hit in zip(finetune_questions, rr_finetune_hits_at_k) if hit == 0] 
```


```python
# Now generate an answer for each question
keyword_search_queries = []
for question in QUESTIONS:
    # Retrieve related documents from retriever
    retriever_results = qa_retriever.retrieve(
        query=question
    )

    # Now generate answer from question and retrieved documents
    predicted_result = qa_generator.predict(
        query=question,
        documents=retriever_results,
        top_k=1
    )

    # Print you answer
    answers = predicted_result["answers"]
#     print(f'Generated answer is \'{answers[0]["answer"]}\' for the question = \'{question}\'')
    keyword_search_queries.append(answers[0]["answer"])
```

    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.95 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.69 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.20 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.02 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.07 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.55 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.41 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.74 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.56 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.13 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.37 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.59 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.81 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.66 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.40 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  7.35 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.52 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.42 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.51 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.64 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.44 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.18 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.44 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.39 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.40 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.61 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.37 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.20 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.57 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.69 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.55 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.33 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.65 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.15 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.84 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.06 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.70 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.46 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.47 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.80 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.23 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.20 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.52 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.78 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.63 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.56 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.23 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.08 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.88 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.13 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.85 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.84 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.03 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.78 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.87 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.68 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.62 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.43 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  7.77 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  7.66 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  7.91 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.46 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.56 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.88 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.48 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.95 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.14 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.01 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.07 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.38 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.69 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.28 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.41 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.69 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.37 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.74 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.04 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.68 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.48 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.86 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.62 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.72 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.70 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.77 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.50 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.56 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.66 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.80 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.88 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.37 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.74 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.47 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.75 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  7.85 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.23 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.43 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.06 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.09 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.63 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.84 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.33 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.36 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.71 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.65 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.17 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.43 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.43 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 14.27 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.68 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.44 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.12 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.35 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.41 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.49 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.15 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.95 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.32 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.21 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.60 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.62 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.75 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.40 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.01 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.74 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.35 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.22 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.99 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.53 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.13 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.49 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.49 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.73 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.27 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.71 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.17 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.02 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.46 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.57 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.50 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.04 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.31 Batches/s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.45 Batches/s]



```python
# Now I fire up the ES doc store to use for BM25 queries using the keywords generated by the step before

# Recommended: Start Elasticsearch using Docker
! docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2
        
# wait until ES has started
! sleep 30
```

    b9bacf69309bb402987d6930b111deb98c0191b5857477057be36dc70a02e899



```python

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

es_document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                            index="document")

```

    01/31/2021 17:59:13 - INFO - elasticsearch -   PUT http://localhost:9200/document [status:200 request:0.401s]
    01/31/2021 17:59:13 - INFO - elasticsearch -   PUT http://localhost:9200/label [status:200 request:0.123s]



```python
from haystack.retriever.sparse import ElasticsearchRetriever

es_retriever = ElasticsearchRetriever(es_document_store)

# from haystack.retriever.sparse import TfidfRetriever

# es_retriever = TfidfRetriever(es_document_store)
```


```python
df2 = df.rename(columns={"text": "question", "answer": "text"})
```


```python
# Convert Dataframe to list of dicts and index them in our DocumentStore
docs_to_index = df2.to_dict(orient="records")

es_document_store.delete_all_documents("document")

es_document_store.write_documents(docs_to_index)
```

    01/31/2021 17:59:14 - INFO - elasticsearch -   POST http://localhost:9200/document/_delete_by_query [status:200 request:0.148s]
    01/31/2021 17:59:15 - INFO - elasticsearch -   POST http://localhost:9200/_bulk?refresh=wait_for [status:200 request:0.491s]



```python
from haystack import Finder

es_finder = Finder(reader=None, retriever=es_retriever)
prediction = es_finder.get_answers_via_similar_questions(question=keyword_search_queries[0], top_k_retriever=3)
print_answers(prediction, details="all")
```

    01/31/2021 23:46:21 - WARNING - haystack.finder -   DEPRECATION WARNINGS: 
                1. The 'Finder' class will be deprecated in the next Haystack release in 
                favour of a new `Pipeline` class that supports building custom search pipelines using Haystack components
                including Retriever, Readers, and Generators.
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/544
                2. The `question` parameter in search requests & results is renamed to `query`.
    01/31/2021 23:46:21 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]


    {   'answers': [   {   'answer': "I studied music and economics. I'm a music "
                                     'major, economics minor and in music I do '
                                     'mostly composition and sound engineering.',
                           'context': "I studied music and economics. I'm a music "
                                      'major, economics minor and in music I do '
                                      'mostly composition and sound engineering.',
                           'document_id': '4d085040-2448-48d6-b397-61efa7f793d6',
                           'meta': {   'id_video': '73bcb9476c0c28cba5c7ddde802b7c63',
                                       'question': 'What do you study?'},
                           'offset_end': 125,
                           'offset_start': 0,
                           'probability': 0.6681556424183887,
                           'question': None,
                           'score': 5.5988407},
                       {   'answer': 'A little bit. My economics minor is helping '
                                     'me with that a lot.',
                           'context': 'A little bit. My economics minor is helping '
                                      'me with that a lot.',
                           'document_id': '9b567d16-0373-4056-a49c-686d8b2f99f1',
                           'meta': {   'id_video': 'b1df3a659a3426c2748cfbeabba5ef28',
                                       'question': 'Is your job related to what '
                                                   'you studied?'},
                           'offset_end': 63,
                           'offset_start': 0,
                           'probability': 0.6457386642329237,
                           'question': None,
                           'score': 4.80288},
                       {   'answer': 'Music completes this artistic and techie '
                                     'side of me because I studied composition and '
                                     'sound engineering, but economics fulfilled '
                                     'my social and analytical side, because I '
                                     'wanted to learn about how to make an impact '
                                     'in the world and I love math.',
                           'context': 'Music completes this artistic and techie '
                                      'side of me because I studied composition '
                                      'and sound engineering, but economics '
                                      'fulfilled my social and analytical side, '
                                      'because I wanted to learn about how to make '
                                      'an impact in the world and I love math.',
                           'document_id': 'be77a3d4-612a-4017-9104-f74f13434fc3',
                           'meta': {   'id_video': '0fbccf3603831fbe3bda2fd760fa7644',
                                       'question': 'Oh, great. So which one do you '
                                                   'like better? Or which one is '
                                                   'like the most appealing to '
                                                   'you?'},
                           'offset_end': 243,
                           'offset_start': 0,
                           'probability': 0.5836848740092451,
                           'question': None,
                           'score': 2.70335}],
        'question': ' economics'}



```python
# %%capture --no-stdout --no-display

# hits_at_1 = 0
# hits_at_k = 0
# hits, probs, scores, answers = [], [], [], []
# for query, question in zip(keyword_search_queries, QUESTIONS):
#     prediction = es_finder.get_answers_via_similar_questions(question=query, top_k_retriever=10);
#     if len(prediction["answers"]) == 0:
#         hits.append(0)
#         hits_at_k += 0
#     else:    
#     answer = prediction['answers'][0]['answer']
#     k_answers = [pred['answer'] for pred in prediction['answers']]
#     if answer in df_dial[df_dial['Q'] == question][['BA1', 'BA2', 'BA3', 'BA4', 'BA5', 'BA6']].values:
#         hits_at_1 += 1
#         hits.append(1)
#     else:
#         hits_at_1 += 0
#         hits.append(0)
#     probs.append(prediction['answers'][0]['probability'])
#     scores.append(prediction['answers'][0]['score'])
#     answers.append(answer)
#     if any([pred_ans in df_dial[df_dial['Q'] == question][['BA1', 'BA2', 'BA3', 'BA4', 'BA5', 'BA6']].values for pred_ans in k_answers]):
#         hits_at_k += 1
#     else:
#         hits_at_k += 0
        
        
        
hits_at_k, hits_at_k_itemized, probs, scores, answers = hitsatk(
    1, 
    es_document_store,
    keyword_search_queries, 
    QUESTIONS,
    df_dial, 
    annotation_cols, 
    finder=es_finder)


```

    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.010s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.011s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.012s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.004s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.009s]
    02/01/2021 09:52:29 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]



```python
print(hits_at_k/len(QUESTIONS))
```

    0.056338028169014086



```python
df_qa_nothit = pd.DataFrame(
{
    "question": QUESTIONS,
    "answer": answers,
    "hit_at_k": hits,
    "prob": probs
})
```


```python
df_qa_nothit.groupby("hit_at_k").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">prob</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>hit_at_k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>117.0</td>
      <td>0.705296</td>
      <td>0.071540</td>
      <td>0.559070</td>
      <td>0.652693</td>
      <td>0.696313</td>
      <td>0.741918</td>
      <td>0.889022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>0.729546</td>
      <td>0.106643</td>
      <td>0.629605</td>
      <td>0.646989</td>
      <td>0.677098</td>
      <td>0.818107</td>
      <td>0.900983</td>
    </tr>
  </tbody>
</table>
</div>




```python
thr_qa = df_qa_nothit.groupby("hit_at_k").describe()["prob"]["min"][1]
thr_qa
```




    0.6296047528376791




```python
j = 10
k = 1
thr = thr_sel

result_items = []
result_probs = []
for question, embedding in zip(finetune_questions, finetune_questions_emb):
    predictions = document_store.query_by_embedding(
        np.array(embedding), 
        top_k=j, 
        return_embedding=False
    )
    pred_answers = [pred.meta['answer'] for pred in predictions]
    qq_probs = np.array([pred.probability for pred in predictions])
    qa_probs = np.array([valid_preds[(valid_preds['q']==question) &
                            (valid_preds['A']==pred_ans)]['y_pred'].values[0] for 
                pred_ans in pred_answers])
    comb_probs = qq_probs * qa_probs        
    sorted_probs = np.sort((comb_probs))[::-1][:k]
    sorted_indices = np.argsort((comb_probs))[::-1][:k]
    pred_answers_reranked = [pred_answers[i] for i, p in zip(sorted_indices, sorted_probs) if p >= thr]
#     print(question, "\n", pred_answers_reranked, "\n")
    if len(pred_answers_reranked) == 0:
        # Retrieve related documents from retriever
        qa_retriever_results = qa_retriever.retrieve(
            query=question
        )
        # Now generate answer from question and retrieved documents
        qa_predicted_result = qa_generator.predict(
            query=question,
            documents=qa_retriever_results,
            top_k=1
        )
        # Get you answer
        answers = qa_predicted_result["answers"]
        keywords_query = answers[0]["answer"]
        prediction = es_finder.get_answers_via_similar_questions(question=keywords_query, top_k_retriever=1)
        if len(prediction["answers"]) > 0:
            aa_prob = prediction["answers"][0]["probability"]
#             print(aa_prob, "\n")
            if aa_prob >= thr_qa:
                pred_answers_reranked.append(prediction["answers"][0]["answer"])
                comb_probs = [aa_prob]
#     print(pred_answers_reranked, "\n==========================================\n")

    annotated_answers = df_dial[df_dial['Q'] == question][annotation_cols].values
    if any([pred_ans in annotated_answers for pred_ans in pred_answers_reranked]):
        result_items.append(1)
        result_probs.append(sorted_probs[0])
    elif (len(pred_answers_reranked) == 0) & (isNaN(annotated_answers[0][0])):
        result_items.append(1)
        result_probs.append(1)
    else:
        result_items.append(0)
        result_probs.append(max(comb_probs))
```

    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.31 Batches/s]
    02/01/2021 12:18:10 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.07 Batches/s]
    02/01/2021 12:18:17 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.73 Batches/s]
    02/01/2021 12:18:26 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.69 Batches/s]
    02/01/2021 12:18:35 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.19 Batches/s]
    02/01/2021 12:18:42 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.55 Batches/s]
    02/01/2021 12:18:51 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.49 Batches/s]
    02/01/2021 12:19:01 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.014s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.19 Batches/s]
    02/01/2021 12:19:15 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.014s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  1.47 Batches/s]
    02/01/2021 12:19:31 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.012s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.28 Batches/s]
    02/01/2021 12:19:43 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.019s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.29 Batches/s]
    02/01/2021 12:19:55 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.012s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.96 Batches/s]
    02/01/2021 12:20:06 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.018s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.44 Batches/s]
    02/01/2021 12:20:17 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.75 Batches/s]
    02/01/2021 12:20:26 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 13.57 Batches/s]
    02/01/2021 12:20:36 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 12.12 Batches/s]
    02/01/2021 12:20:47 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.013s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.54 Batches/s]
    02/01/2021 12:20:57 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.96 Batches/s]
    02/01/2021 12:21:07 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.62 Batches/s]
    02/01/2021 12:21:17 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.30 Batches/s]
    02/01/2021 12:21:26 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.84 Batches/s]
    02/01/2021 12:21:36 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.44 Batches/s]
    02/01/2021 12:21:46 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.88 Batches/s]
    02/01/2021 12:21:57 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  6.82 Batches/s]
    02/01/2021 12:22:06 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.60 Batches/s]
    02/01/2021 12:22:15 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  5.11 Batches/s]
    02/01/2021 12:22:23 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.19 Batches/s]
    02/01/2021 12:22:31 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.39 Batches/s]
    02/01/2021 12:22:41 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.83 Batches/s]
    02/01/2021 12:22:51 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.98 Batches/s]
    02/01/2021 12:22:59 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.01 Batches/s]
    02/01/2021 12:23:10 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.91 Batches/s]
    02/01/2021 12:23:18 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.45 Batches/s]
    02/01/2021 12:23:27 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.46 Batches/s]
    02/01/2021 12:23:36 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.61 Batches/s]
    02/01/2021 12:23:45 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.36 Batches/s]
    02/01/2021 12:23:55 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.39 Batches/s]
    02/01/2021 12:24:04 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.36 Batches/s]
    02/01/2021 12:24:13 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.22 Batches/s]
    02/01/2021 12:24:22 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.85 Batches/s]
    02/01/2021 12:24:31 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.52 Batches/s]
    02/01/2021 12:24:40 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.06 Batches/s]
    02/01/2021 12:24:49 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.09 Batches/s]
    02/01/2021 12:24:59 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.85 Batches/s]
    02/01/2021 12:25:09 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.60 Batches/s]
    02/01/2021 12:25:18 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.22 Batches/s]
    02/01/2021 12:25:26 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.59 Batches/s]
    02/01/2021 12:25:36 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.77 Batches/s]
    02/01/2021 12:25:45 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.10 Batches/s]
    02/01/2021 12:25:55 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.82 Batches/s]
    02/01/2021 12:26:07 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.41 Batches/s]
    02/01/2021 12:26:15 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.27 Batches/s]
    02/01/2021 12:26:24 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.61 Batches/s]
    02/01/2021 12:26:33 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.77 Batches/s]
    02/01/2021 12:26:43 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.29 Batches/s]
    02/01/2021 12:26:51 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  6.88 Batches/s]
    02/01/2021 12:27:00 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.97 Batches/s]
    02/01/2021 12:27:10 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.94 Batches/s]
    02/01/2021 12:27:19 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.009s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.37 Batches/s]
    02/01/2021 12:27:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.004s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.53 Batches/s]
    02/01/2021 12:27:37 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.30 Batches/s]
    02/01/2021 12:27:46 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.68 Batches/s]
    02/01/2021 12:27:55 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.68 Batches/s]
    02/01/2021 12:28:03 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.10 Batches/s]
    02/01/2021 12:28:14 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.06 Batches/s]
    02/01/2021 12:28:23 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.36 Batches/s]
    02/01/2021 12:28:32 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.70 Batches/s]
    02/01/2021 12:28:41 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.009s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.64 Batches/s]
    02/01/2021 12:28:52 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  7.92 Batches/s]
    02/01/2021 12:29:02 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.11 Batches/s]
    02/01/2021 12:29:11 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.00 Batches/s]
    02/01/2021 12:29:20 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.008s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.63 Batches/s]
    02/01/2021 12:29:28 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.21 Batches/s]
    02/01/2021 12:29:38 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.027s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  8.87 Batches/s]
    02/01/2021 12:29:48 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.62 Batches/s]
    02/01/2021 12:29:58 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.14 Batches/s]
    02/01/2021 12:30:08 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00,  9.86 Batches/s]
    02/01/2021 12:30:16 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.007s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.13 Batches/s]
    02/01/2021 12:30:27 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.38 Batches/s]
    02/01/2021 12:30:36 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.005s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 10.75 Batches/s]
    02/01/2021 12:30:45 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.011s]
    Creating Embeddings: 100%|██████████| 1/1 [00:00<00:00, 11.09 Batches/s]
    02/01/2021 12:30:54 - INFO - elasticsearch -   POST http://localhost:9200/document/_search [status:200 request:0.006s]



```python
print(sum(result_items)/len(finetune_questions))

# without thresholding it was 0.175
# 0.2125 on test questions
```

    0.1949685534591195



```python
df_qa_finetune = pd.DataFrame(
{
    "question": finetune_questions,
    "hit_at_k": result_items,
    "prob": result_probs
})
```


```python
df_qa_finetune.to_csv("~/Documents/df_qa_finetune.csv")
```


```python
rr_finetune = pd.DataFrame(
{
    "question": finetune_questions,
    "hit_at_k": rr_finetune_hits_at_k,
    "prob": rr_finetune_probs
})
rr_finetune.to_csv("~/Documents/rr_finetune.csv")
```


```python

```
