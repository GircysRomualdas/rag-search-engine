import os

import numpy as np
from lib.utils.constants import (
    CACHE_CHUNK_EMBEDDINGS_PATH,
    CHCHE_CHUNK_METADATA_PATH,
    DEFAULT_MODEL,
)
from lib.utils.load import load_chunk_metadata
from lib.utils.write import write_json

from .semantic_search import SemanticSearch, get_sentences_chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray = None
        self.chunk_metadata: list[dict] = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        all_chunks: list[str] = []
        chunks_metadata: list[dict] = []
        for doc_idx, document in enumerate(documents):
            self.document_map[document["id"]] = document

            if len(document["description"]) == 0:
                continue

            chunks = get_sentences_chunks(
                text=document["description"], chunk_size=4, overlap=1
            )
            all_chunks.extend(chunks)

            for chu_idx, chunk in enumerate(chunks):
                chunks_metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chu_idx,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata

        np.save(CACHE_CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        write_json(
            path=CHCHE_CHUNK_METADATA_PATH,
            data={"chunks": chunks_metadata, "total_chunks": len(all_chunks)},
        )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(CACHE_CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHCHE_CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CACHE_CHUNK_EMBEDDINGS_PATH)
            self.chunk_metadata = load_chunk_metadata()

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
