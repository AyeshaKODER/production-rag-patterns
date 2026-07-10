import os
import re
from src.embedders.base_embedder import BaseEmbedder
from src.llms.base_llm import BaseLLM
from src.vectorstores.base_vectorstore import BaseVectorStore
from src.rerankers.base_reranker import BaseReranker


class RAGPipeline:
    """
    Ties embedder + vector store + LLM together into ingest -> retrieve -> generate,
    with an optional re-ranking stage between retrieval and generation.

    Depends only on the abstract base classes, not concrete providers —
    swapping LocalEmbedder for OpenAIEmbedder, GroqLLM for another LLM, or
    adding/removing a reranker, requires zero changes here.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
        llm: BaseLLM,
        reranker: BaseReranker = None,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        self.reranker = reranker  # optional — pipeline works fine without one

    # ---------- Ingestion ----------

    def ingest_directory(self, dir_path: str, max_chunk_chars: int = 800) -> int:
        """
        Recursively reads every supported file under dir_path (.pdf, .docx,
        .pptx, .html, .txt), chunks it, embeds each chunk, and adds it to
        the vector store. Returns total chunk count ingested.
        """
        from src.ingestion.extractors import extract_text

        total_chunks = 0
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if filename.startswith("~$") or filename.startswith("."):
                    continue  # skip Office lock files and hidden files

                filepath = os.path.join(root, filename)
                text = extract_text(filepath)

                if text is None:
                    continue  # unsupported extension, skip silently

                chunks = self._chunk_text(text, max_chunk_chars)
                if not chunks:
                    continue

                embeddings = [self.embedder.embed(chunk) for chunk in chunks]
                self.vectorstore.add(texts=chunks, embeddings=embeddings)
                total_chunks += len(chunks)

        return total_chunks

    # ---------- Chunking ----------

    def _chunk_text(self, text: str, max_chunk_chars: int, min_chunk_chars: int = 80) -> list[str]:
        """
        Section-aware chunking: splits on markdown-style headers (##) and
        blank-line paragraph breaks first, so each chunk stays semantically
        whole (a full policy section, not a sentence cut in half).

        Any resulting section still longer than max_chunk_chars gets
        further split on paragraph breaks as a fallback, so one giant
        section can't blow up retrieval with an oversized chunk.

        Sections shorter than min_chunk_chars are dropped — a lone title
        line or bare header with no real content underneath (e.g. "##
        Slide 2" immediately followed by the next header) is noise, not a
        retrievable unit, and would otherwise pollute retrieval results
        with near-empty context.
        """
        text = self._strip_leading_title(text)
        sections = re.split(r'\n(?=#{1,3}\s)', text)

        chunks = []
        for section in sections:
            section = section.strip()
            if not section or len(section) < min_chunk_chars:
                continue

            if len(section) <= max_chunk_chars:
                chunks.append(section)
            else:
                paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
                buffer = ""
                for para in paragraphs:
                    if len(buffer) + len(para) <= max_chunk_chars:
                        buffer += ("\n\n" + para if buffer else para)
                    else:
                        if buffer and len(buffer) >= min_chunk_chars:
                            chunks.append(buffer)
                        buffer = para
                if buffer and len(buffer) >= min_chunk_chars:
                    chunks.append(buffer)

        return chunks

    @staticmethod
    def _strip_leading_title(text: str, max_title_chars: int = 100) -> str:
        """
        Removes a standalone title first line (short, no sentence-ending
        punctuation) before chunking, so it doesn't ride along at the front
        of chunk 1 and make answers look like they're echoing a headline.
        """
        lines = text.split("\n", 1)
        if not lines:
            return text
        first_line = lines[0].strip()
        looks_like_title = (
            0 < len(first_line) <= max_title_chars
            and not first_line.endswith((".", "?", "!"))
        )
        if looks_like_title and len(lines) > 1:
            return lines[1].lstrip("\n")
        return text
    
    # ---------- Retrieval + Generation ----------

    def query(
        self,
        question: str,
        top_k: int = 3,
        retrieval_pool_size: int = 10,
        return_context: bool = False,
    ):
        """
        Embeds the question, retrieves a wider candidate pool
        (retrieval_pool_size), re-ranks it down to top_k if a reranker is
        configured, and asks the LLM to answer using only that context.

        retrieval_pool_size only matters when a reranker is set — without
        one, the pipeline retrieves top_k directly as before.

        If return_context=True, returns (answer, retrieved_chunks) instead
        of just the answer string — used by the UI and eval scripts that
        need to show/inspect what was actually retrieved.
        """
        query_embedding = self.embedder.embed(question)

        if self.reranker:
            candidates = self.vectorstore.query(query_embedding, top_k=retrieval_pool_size)
            retrieved_chunks = self.reranker.rerank(question, candidates, top_k=top_k)
        else:
            retrieved_chunks = self.vectorstore.query(query_embedding, top_k=top_k)

        context = "\n\n---\n\n".join(retrieved_chunks)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        answer = self.llm.generate(prompt)

        if return_context:
            return answer, retrieved_chunks
        return answer