# RAG search engine

A semantic search engine using RAG and LLMs to power intelligent movie search.
It includes classic keyword ranking, semantic embeddings, and basic RAG-style chunking utilities.

This is the starter code used in Boot.dev's [Learn Retrieval Augmented Generation](https://www.boot.dev/courses/learn-retrieval-augmented-generation) course.

---

## Features

- Classic keyword search with TF, IDF, TF‑IDF, and BM25 scoring
- Semantic search over movie descriptions using sentence embeddings
- Chunked (sentence-based) semantic search for RAG-style retrieval
- Hybrid search combining BM25 and semantic scores with a tunable α

---

## Requirements
- Python 3.12+
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

4. Create a file at `data/movies.json` and populate it with the JSON movie data from [movie dataset file](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json)

5. Create a file at `data/stopwords.txt` and populate it with stop words from [this list of stop words](https://countwordsfree.com/stopwords)

---

## Usage

### Keyword search

#### Keyword search commands
| Command                                   | Purpose                         |
|-------------------------------------------|---------------------------------|
| `build`                                   | Parse `movies.json` and build the cached inverted index and doc map.|
| `search <query>`                          | Keyword search ranked by raw term frequency.|
| `tf <doc_id> <term>`                      | Show how many times `<term>` appears in document `<doc_id>`.|
| `idf <term>`                              | Show how rare `<term>` is across all documents (inverse doc frequency).|
| `tfidf <doc_id> <term>`                   | Compute the TF‑IDF relevance score of `<term>` in `<doc_id>`.|
| `bm25idf <term>`                          | Show the BM25‑style IDF value for `<term>`.|
| `bm25tf <doc_id> <term> [k1] [b]`         | Compute the BM25 term-frequency score for `<term>` in `<doc_id>`.|
| `bm25search "<query>" [limit]`            | Full BM25 ranking search over all documents for `<query>`.|

---

#### Build command:
```bash
uv run cli/keyword_search_cli.py build
```

Output:
```
Building inverted index...
Inverted index written to cache/index.pkl
Document map written to cache/docmap.pkl
Inverted index built successfully.
```

---

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
```
Searching for: the hot shot
1. The Chronicles of Narnia: Prince Caspian
2. 10
3. Bombai Ka Babu
4. Hot Potato
5. Beerfest
```

---

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
```
Term frequency of 'trapper' in document 424: 4
```

---

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
```
Inverse document frequency of 'grizzly': 5.52
```

---

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
```
TF-IDF score of 'trapper' in document '424': 24.13
```

---

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
```
BM25 IDF score of 'grizzly': 5.55
```

---

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
```
BM25 TF score of 'anbuselvan' in document '1': 2.35
```

---

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
```
Searching for: space adventure
1. (1763) Space Ace - Score: 10.65
2. (841) The Adventures of the Galaxy Rangers - Score: 9.53
3. (4946) Titanfall - Score: 9.12
4. (4430) House II: The Second Story - Score: 7.27
5. (3216) Les ma\u00eetres du temps - Score: 6.85
```

---

### Semantic search

#### Semantic search commands
| Command                                             | Purpose                               |
|-----------------------------------------------------|---------------------------------------|
| `verify`                                                    | Load the embedding model and print basic model and config information.|
| `embed_text <text>`                                         | Generate and inspect an embedding vector for arbitrary `<text>`.|
| `verify_embeddings`                                         | Verify that precomputed movie embeddings exist and match the dataset.|
| `embedquery <query>`                                        | Embed a natural-language `<query>` for use in semantic search.|
| `search <query> [--limit N]`                                | Rank movies by cosine similarity between `<query>` and movie embeddings.|
| `chunk <text> [--chunk-size N] [--overlap M]`               | Split long `<text>` into word chunks, optionally with an overlap of `M` words between consecutive chunks.|
| `semantic_chunk <text> [--max-chunk-size N] [--overlap M]`  | Split long `<text>` into sentence-based semantic chunks, optionally with sentence overlap. |
| `embed_chunks`                                              | Create and store embeddings for document chunks. |
| `search_chunked <query> [--limit N]`                        | Run semantic search over chunk embeddings and aggregate scores back to the most relevant movies. |

---

#### Verify command:
```bash
uv run cli/semantic_search_cli.py verify
```

Output:
```
Model loaded: SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
Max sequence length: 256
```

---

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
```
Text: Luke, I am your father
First 3 dimensions: [-0.03583722 -0.01693317  0.04318329]
Dimensions: 384
```

---

#### Verify movie embeddings command:
```bash
uv run cli/semantic_search_cli.py verify_embeddings
```

Output:
```
Number of docs:   5000
Embeddings shape: 5000 vectors in 384 dimensions
```

---

#### Embed query command:
```bash
uv run cli/semantic_search_cli.py embedquery <query>
```

- `<query>`: the search query you want to generate an embedding for.

##### Example
```bash
uv run cli/semantic_search_cli.py embedquery "funny bear movies"
```

Output:
```
Query: funny bear movies
First 5 dimensions: [-0.07288318 -0.01480833  0.00170603  0.07981379  0.01246783]
Shape: (384,)
```

---

#### Search command:
```bash
uv run cli/semantic_search_cli.py search <query> [--limit N]
```

- `<query>`: the natural language query you want to search for.
- `--limit N` (optional): maximum number of results to return (default is 5).

##### Example
```bash
uv run cli/semantic_search_cli.py search "funny bear movies" --limit 5
```

Output:
```
1. Bear (score: 0.5590)
   Businessman Sam, his wife Liz and his musician brother Nick with his girlfriend Christine are drivin...

2. The Great Bear (score: 0.5104)
   Jonathan (11 years old) is playing hide in seek with his younger sister Sophie ( 6 years old ), Soph...

3. A Bear for Punishment (score: 0.4949)
   The film begins with the bear family sleeping peacefully at home, when suddenly, the alarms of dozen...

4. An Unfinished Life (score: 0.4945)
   One year ago, a wild bear stole a calf from Mitch (Morgan Freeman) and Einar\u2019s (Robert Redford)...

5. Unnatural (score: 0.4904)
   Deep in the remote wilderness of Alaska, a secretive biotechnology corporation has been conducting i...
```

---

#### Chunk command:
```bash
uv run cli/semantic_search_cli.py chunk <text> [--chunk-size N] [--overlap M]
```

- `<text>`: the full text you want to split into chunks.
- `--chunk-size N` (optional): number of words per chunk (defaults to 200).
- `--overlap M` (optional): number of words to share between consecutive chunks (defaults to 0, meaning no overlap).

##### Example
```bash
uv run cli/semantic_search_cli.py chunk "This is a test text with two chunks" --chunk-size 5 --overlap 2
```

Output:
```
Chunking 35 characters
1. This is a test text
2. test text with two chunks
```

---

#### Semantic chunk command:
```bash
uv run cli/semantic_search_cli.py semantic_chunk <text> [--max-chunk-size N] [--overlap M]
```

- `<text>`: the full text you want to split into semantic (sentence-based) chunks.
- `--max-chunk-size N` (optional): maximum number of sentences per chunk (defaults to 4).
- `--overlap M` (optional): number of sentences to share between consecutive chunks (defaults to 0, meaning no overlap).

##### Example
```bash
uv run cli/semantic_search_cli.py semantic_chunk "First sentence here. Second sentence here. Third sentence here. Fourth sentence here." --max-chunk-size 2 --overlap 1
```

Output:
```
Semantically chunking 85 characters
1. First sentence here. Second sentence here.
2. Second sentence here. Third sentence here.
3. Third sentence here. Fourth sentence here.
```

---

#### Embed chunks command:
```bash
uv run cli/semantic_search_cli.py embed_chunks
```

Output:
```
Generated 72909 chunked embeddings
```

---

#### Chunked semantic search command:
```bash
uv run cli/semantic_search_cli.py search_chunked <query> [--limit N]
```

- `<query>`: the natural-language search query you want to run against the chunked movie descriptions.
- `--limit N` (optional): maximum number of movies to return (defaults to 5).

##### Example
```bash
uv run cli/semantic_search_cli.py search_chunked "superhero action movie" --limit 5
```

Output:
```
1. Kick-Ass (score: 0.6386)
   Dave Lizewski (Aaron Taylor-Johnson) opens the film with a narration about how superheroes came into...

2. The Incredibles (score: 0.5386)
   The film opens with a series of short interviews between three famous superheroes including Mr. Incr...

3. Logan (score: 0.5280)
   The film is preceded by a short film:On the streets of New York City, a mugging is taking place. Wal...

4. Justice League: The Flashpoint Paradox (score: 0.5144)
   While visiting his mother's grave, Barry Allen, known as the Flash, is alerted to a break-in by Capt...

5. Justice League: Throne of Atlantis (score: 0.5117)
   In the Atlantic Ocean, the USS California submarine picks up incoming human-like threats approaching...
```

---

### Hybrid search

#### Hybrid search commands

| Command                                                     | Purpose                                                                                  |
|-------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `normalize <score1> <score2> ...`                           | Normalize a list of numeric scores to the 0–1 range (min–max scaling).                  |
| `weighted-search <query> [--alpha A] [--limit N]`           | Rank movies using a weighted combination of BM25 (keyword) and semantic scores.         |
| `rrf-search <query> [-k K] [--limit N] [--enhance METHOD]`       | Rank movies using Reciprocal Rank Fusion, optionally enhancing the query first.         |

---

#### Normalize command:
```bash
uv run cli/hybrid_search_cli.py normalize <score1> <score2> ...
```

- `<score1> <score2> ...`: one or more numeric scores (floats) to normalize.

##### Example
```bash
uv run cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1
```

Output:
```
* 0.1818
* 1.0000
* 0.5000
* 0.1818
* 0.0000
```

---

#### Weighted hybrid search command
```bash
uv run cli/hybrid_search_cli.py weighted-search <query> [--alpha A] [--limit N]
```

- `<query>`: the natural-language search query you want to run.
- `--alpha A` (optional): weighting factor between keyword and semantic scores.
  - `A = 1.0` → 100% BM25 (keyword-only)
  - `A = 0.0` → 100% semantic
  - defaults to `0.5` (50/50 hybrid)
- `--limit N` (optional): maximum number of movies to return (defaults to `5`).

##### Example
```bash
uv run cli/hybrid_search_cli.py weighted-search "British Bear" --alpha 0.5 --limit 5
```

Output:
```
1. Paddington
   Hybrid Score: 0.966
   BM25: 1.000, Semantic: 0.932
   Deep in the rainforests of Peru, a young bear lives peacefully with his Aunt Lucy and Uncle Pastuzo,
2. Legends of the Fall
   Hybrid Score: 0.740
   BM25: 0.480, Semantic: 1.000
   Sick of betrayals the United States government perpetrated on the Native Americans, Colonel William 
3. The Country Bears
   Hybrid Score: 0.708
   BM25: 0.604, Semantic: 0.812
   Beary Barrington is a young bear who has been raised by a human family and struggles with his identi
4. The Edge
   Hybrid Score: 0.701
   BM25: 0.567, Semantic: 0.836
   Charles Morse, an intellectual billionaire with a vast knowledge of survival techniques, accompanies
5. The Great Bear
   Hybrid Score: 0.692
   BM25: 0.624, Semantic: 0.760
   Jonathan (11 years old) is playing hide in seek with his younger sister Sophie ( 6 years old ), Soph
```

---

#### Reciprocal Rank Fusion (RRF) search command
```bash
uv run cli/hybrid_search_cli.py rrf-search <query> [-k K] [--limit N] [--enhance METHOD]
```

- `<query>`: the natural-language search query you want to run.
- `-k K` (optional): RRF `k` parameter that controls how quickly the reciprocal scores decay with rank.
  - smaller `K` means a steeper decay and more weight on top-ranked results
  - larger `K` means a flatter decay and more influence from lower-ranked results
  - defaults to `60`
- `--limit N` (optional): maximum number of movies to return (defaults to `5`).
- `--enhance METHOD` (optional): apply a query enhancement method before running RRF search.
  - `spell`: fix obvious spelling mistakes in the query via an LLM, preserving correctly spelled words.
  - `rewrite`: rewrite the query to be more specific and searchable.

##### Example
```bash
uv run cli/hybrid_search_cli.py rrf-search "briish bear" --limit 5 --enhance spell
```

Output:
```
Enhanced query (spell): 'briish bear' -> 'Corrected: "british bear'

1. Paddington
   RRF Score: 0.032
   BM25 Rank: 3, Semantic Rank: 2
   Deep in the rainforests of Peru, a young bear lives peacefully with his Aunt Lucy and Uncle Pastuzo,...

2. The Great Bear
   RRF Score: 0.028
   BM25 Rank: 20, Semantic Rank: 3
   Jonathan (11 years old) is playing hide in seek with his younger sister Sophie ( 6 years old ), Soph...

3. The Country Bears
   RRF Score: 0.026
   BM25 Rank: 30, Semantic Rank: 7
   Beary Barrington is a young bear who has been raised by a human family and struggles with his identi...

4. The Bear
   RRF Score: 0.025
   BM25 Rank: 24, Semantic Rank: 16
   Tilly is a young girl who treasures her beloved teddy bear above all other possessions, carrying it ...

5. Bear
   RRF Score: 0.025
   BM25 Rank: 28, Semantic Rank: 15
   Businessman Sam, his wife Liz and his musician brother Nick with his girlfriend Christine are drivin...
```
