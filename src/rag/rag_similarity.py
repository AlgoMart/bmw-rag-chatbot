from pathlib import Path
from typing import List

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


class VectorDBService:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_DB_COLLECTION_NAME = "bmw_descriptions"
    CHROMA_DB_DIRECTORY_NAME = "./chroma_db"

    def load_description(self) -> List[str]:
        # Load the data
        df = pd.read_csv("data/rag_data.csv")

        # Check the description column availability
        if "DESCRIPTION" not in df.columns:
            return "DESCRIPTION column is not available"

        # Check if the description column contains any rows ot not
        if df["DESCRIPTION"].shape == 0:
            return "DESCRIPTION column does not have any rows"

        # Return the rows of description column
        return df["DESCRIPTION"].values.tolist()

    def store_documents_to_the_vector_db(self) -> Chroma:
        # Load the OpenAIEmbedding
        embedding_model = HuggingFaceEmbeddings(
            model_name=VectorDBService.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        # Intialize the ChromaDB
        vector_store = Chroma.from_texts(
            texts=self.load_description(),
            collection_name=VectorDBService.CHROMA_DB_COLLECTION_NAME,
            embedding=embedding_model,
            persist_directory=VectorDBService.CHROMA_DB_DIRECTORY_NAME,
        )

        # Return the ventor store instance
        return vector_store

    def get_ventor_store_retriever(self, number_of_documents: int) -> VectorStoreRetriever:
        # Load the OpenAIEmbedding
        embedding_model = HuggingFaceEmbeddings(
            model_name=VectorDBService.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        # Initialize the ventor store instance
        vector_store = Chroma(
            collection_name=VectorDBService.CHROMA_DB_COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=VectorDBService.CHROMA_DB_DIRECTORY_NAME,
        )

        # Initialize the retriever and return it
        return vector_store.as_retriever(search_kwargs={"k": number_of_documents})

    def query_ventor_store_retriever_with_score(
        self, query: str, number_of_documents: int
    ) -> List[tuple[Document, float]]:
        # Load the OpenAIEmbedding
        embedding_model = HuggingFaceEmbeddings(
            model_name=VectorDBService.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        # Initialize the ventor store instance
        vector_store = Chroma(
            collection_name=VectorDBService.CHROMA_DB_COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=VectorDBService.CHROMA_DB_DIRECTORY_NAME,
        )

        # Initialize the retriever and return it
        return vector_store.similarity_search_with_score(query, k=number_of_documents)

    def query_results(self, query: str | None, number_of_documents: int = 6, check_with_scores: bool = True) -> None:
        # Use default query if none provided
        if query is None:
            query = (
                "indicator red fast movement 1.6a 250v holder plastic 5 x 20mm ceramic box "
                "ccc/pse/vde/culus electric indicator, very fast blow, 1.6a, 250vac, 1500a (ir), "
                "inline/holder, 5x20mm"
            )

        if not check_with_scores:
            retriever = self.get_ventor_store_retriever(number_of_documents=number_of_documents)
            results = retriever.invoke(query)

            print(f"\nTop {number_of_documents} Documents (No similarity scores):\n")
            for i, result in enumerate(results):
                print(f"Result {i + 1}:")
                print(f"Document: {result.page_content}")
                print("-" * 80)

        else:
            results = self.query_ventor_store_retriever_with_score(
                query=query, number_of_documents=number_of_documents
            )

            print(f"\nTop {number_of_documents} Documents with Similarity Scores:\n")
            for i, (doc, score) in enumerate(results):
                similarity_percentage = (1 - score) * 100  # Assuming cosine distance
                print(f"Result {i + 1}:")
                print(f"Document: {doc.page_content}")
                print(f"Similarity: {similarity_percentage:.2f}%")
                print("-" * 80)


if __name__ == "__main__":
    vector_store_path = Path("chroma_db")

    vector_db_service = VectorDBService()
    if not vector_store_path.exists():
        vector_db_service.store_documents_to_the_vector_db()
        print("Successfully stored BMW data into vector db!")

    vector_db_service.query_results(query=None)
