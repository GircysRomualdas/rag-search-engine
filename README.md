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

### Keywird search

#### Build command:
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

#### Search command:
```bash
uv run cli/keyword_search_cli.py search <query>
```

- `<query>`: the text you want to search for.

##### Example
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

#### Term frequency (TF) command:
```bash
uv run cli/keyword_search_cli.py tf <doc_id> <term>
```

- `<doc_id>`: the ID of the document.
- `<term>`: the term whose frequency you want to retrieve.

##### Example
```bash
uv run cli/keyword_search_cli.py tf 424 trapper
```

Output:
```bash
Term frequency of 'trapper' in document 424: 4
```

#### Inverse document frequency (IDF) command:
```bash
uv run cli/keyword_search_cli.py idf <term>
```

- `<term>`: the term for which you want to calculate the Inverse Document Frequency.

##### Example
```bash
uv run cli/keyword_search_cli.py idf grizzly
```

Output:
```bash
Inverse document frequency of 'grizzly': 5.52
```

#### Term Frequency-Inverse Document Frequency (TF-IDF) command:
```bash
uv run cli/keyword_search_cli.py tfidf <doc_id> <term>
```

- `<doc_id>`: the ID of the document to score.
- `<term>`: the term for which you want to calculate the TF-IDF score.

##### Example
```bash
uv run cli/keyword_search_cli.py tfidf 424 trapper
```

Output:
```bash
TF-IDF score of 'trapper' in document '424': 24.13
```

#### BM25 Inverse Document Frequency (BM25 IDF) command:
```bash
uv run cli/keyword_search_cli.py bm25idf <term>
```

- `<term>`: the term for which you want to calculate the BM25 IDF score.

##### Example
```bash
uv run cli/keyword_search_cli.py bm25idf grizzly
```

Output:
```bash
BM25 IDF score of 'grizzly': 5.55
```

#### BM25 Term Frequency (BM25 TF) command:
```bash
uv run cli/keyword_search_cli.py bm25tf <doc_id> <term> [k1]
```

- `<doc_id>`: the document ID in which you want to calculate the BM25 TF score.
- `<term>`: the term for which you want to calculate the BM25 TF score.
- `[k1]` (optional): the BM25 saturation parameter, defaults to 1.5.
- `[b]` (optional): the BM25 length-normalization parameter, defaults to 0.75

##### Example
```bash
uv run cli/keyword_search_cli.py bm25tf 1 anbuselvan
```

Output:
```bash
BM25 TF score of 'anbuselvan' in document '1': 2.35
```

#### BM25 Search (BM25 full scoring) command:
```bash
uv run cli/keyword_search_cli.py bm25search "<query>" [limit]
```

- `<query>`: the search query string (can contain multiple terms, so it’s best to quote it).
- `[limit]` (optional): the maximum number of search results to return. Defaults to 5.

##### Example
```bash
uv run cli/keyword_search_cli.py bm25search "space adventure"
```

Output:
```bash
Searching for: space adventure
1. (1763) Space Ace - Score: 10.65
2. (841) The Adventures of the Galaxy Rangers - Score: 9.53
3. (4946) Titanfall - Score: 9.12
4. (4430) House II: The Second Story - Score: 7.27
5. (3216) Les ma\u00eetres du temps - Score: 6.85
```

---

### Semantic search

#### Verify command:
```bash
uv run cli/semantic_search_cli.py verify
```

Output:
```bash
Model loaded: SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
Max sequence length: 256
```

#### Embed text command:
```bash
uv run cli/semantic_search_cli.py embed_text <text>
```

- `<text>`: the text you want to generate an embedding for.

##### Example
```bash
uv run cli/semantic_search_cli.py embed_text "Luke, I am your father"
```

Output:
```bash
Text: Luke, I am your father
First 3 dimensions: [-0.03583722 -0.01693317  0.04318329]
Dimensions: 384
```
