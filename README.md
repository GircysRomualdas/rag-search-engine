# RAG search engine

A semantic search engine using RAG and LLMs to power intelligent movie search.

This is the starter code used in Boot.dev's [Learn Retrieval Augmented Generation](https://www.boot.dev/courses/learn-retrieval-augmented-generation) course.

---

## Requirements
- Python 3
- uv (if not installed: `pip install uv`)

---

## Installation

1. Clone the repository.

2. Create and activate a virtual environment:
```bash
uv venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Create a file at `data/movies.json` and populate it with the JSON movie data from [movie dataset](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json)

5. Create a file at `data/stopwords.txt` and populate it with stop words from [The list of stop words](https://countwordsfree.com/stopwords)

---

## Usage

### Build command:
```bash
uv run cli/keyword_search_cli.py build
```

Output:
```bash
Building inverted index...
Inverted index written to cache/index.pkl
Document map written to cache/docmap.pkl
Inverted index built successfully.
```

### Search command:
```bash
uv run cli/keyword_search_cli.py search <query>
```

- `<query>`: the text you want to search for.

#### Example
```bash
uv run cli/keyword_search_cli.py search "the hot shot"
```

Output:
```bash
Searching for: the hot shot
1. Hot Potato
2. Hot Shots! Part Deux
3. Hotel Chevalier
4. Hotel Berlin
5. Killshot
```

### Term frequency (TF) command:
```bash
uv run cli/keyword_search_cli.py tf <doc_id> <term>
```

- `<doc_id>`: the ID of the document.
- `<term>`: the term whose frequency you want to retrieve.

#### Example
```bash
uv run cli/keyword_search_cli.py tf 424 trapper
```

Output:
```bash
Term frequency of 'trapper' in document 424: 4
```

### Inverse document frequency (IDF) command:
```bash
uv run cli/keyword_search_cli.py idf <term>
```

- `<term>`: the term for which you want to calculate the Inverse Document Frequency.

#### Example
```bash
uv run cli/keyword_search_cli.py idf grizzly
```

Output:
```bash
Inverse document frequency of 'grizzly': 5.52
```

### Term Frequency-Inverse Document Frequency (TF-IDF) command:
```bash
uv run cli/keyword_search_cli.py tfidf <doc_id> <term>
```

- `<doc_id>`: the ID of the document to score.
- `<term>`: the term for which you want to calculate the TF-IDF score.

#### Example
```bash
uv run cli/keyword_search_cli.py tfidf 424 trapper
```

Output:
```bash
TF-IDF score of 'trapper' in document '424': 24.13
```
