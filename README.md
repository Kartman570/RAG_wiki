Local RAG on wikipedia dataset.



### Installation
```
poetry install
```

### Start script
```
streamlit run ./streamlit_app.py
```

First run require download wikipedia dataset (compact version **235 Mb**).<br>
Then dataset going to save vector indexes (**890 Mb**). That require about **8.2 Gb of RAM**.

When calculation is over, dataset and indexes stores locally with no need to recalculate.
Streamlit default address
***localhost:8501***