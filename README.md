# dm_api - Dialogue Manager for TOIA++

For making the Flask app working, create a folder named *faiss_indices*:

```mkdir faiss_indeces```

You need to install PostgreSQL and make sure SQLAlchemy-Utils is version 0.36.7 as per the requirements.txt.

To update the db for including Margarita's avatar, run the first part of the notebook (until "Dialogue Mgr can stop here"). Make sure to use the Margarita avatar id as the db dame!

If you want to use the rest of the notebook, you need to create a folder named *data*,

```mkdir data```,

and save in there the files `MargaritaCorpusKB_video_id.csv` and `DIALOGUES.csv` avaible <a href="https://drive.google.com/drive/folders/1KfPgHZ5NXjKPYAZToYExL6e8DHAxPfC8?usp=sharing" target="_blank" rel="noopener noreferrer">here</a>.

### Aknoledgments ###
Thanks to [Haystack](https://haystack.deepset.ai/). My code was inspired from [Tutorial 4](https://haystack.deepset.ai/docs/latest/tutorial4md).
