from src.ingestion.parser import SimpleLoader
from src.ingestion.chunker import RecursiveChunk
from src.embedder.ollama_embedder import OllamaEmbedder
from src.database.pgvector_storage import PGVectorStore
from src.utils.helpers import IngestionPipeline, RAGChain
from src.generation.ollama_llm import OllamaLLM
from src.reranker.cross_encoder_reranker import CrossEncoderReRanker


if __name__ == "__main__":
    running = True
    while running:
        print("1. Providing documents for ChatBot.")
        print("2. Chat with Bot.")
        print("3. Exit.")
        choice = int(input("Input your choice: "))
        if choice == 3:
            running = False
        elif choice == 1:
            ingestor = IngestionPipeline(SimpleLoader(), RecursiveChunk(1200, 50),
                                         OllamaEmbedder(dimensions=2046, num_ctx=8192, num_gpu=1), PGVectorStore())
            while True:
                print("1. Ingest one pdf file.")
                print("2. Ingest by directory.")
                print("3. Exit.")
                choice = int(input("Input your choice: "))
                if choice == 3:
                    break
                elif choice == 1:
                    input_file = input("Please input your file directory: ")
                    ingestor.ingest_pdf(input_file)
                else:
                    input_dir = input("Please input your directory: ")
                    ingestor.ingest_dir(input_dir)
        else:
            llm = OllamaLLM(temperature=0.1, model_name="qwen3.5:2b", num_ctx=4096, num_gpu=1, num_predict=2046, reasoning=False)
            repository = PGVectorStore()
            model_embed = OllamaEmbedder(dimensions=2046, num_ctx=8192, num_gpu=1)

            reranker = CrossEncoderReRanker()
            rag_chain = RAGChain(llm, repository, model_embed, reranker)

            while True:
                user = input("Input your question (exit for exit program): ")
                if user.lower() == "exit":
                    break
                response = rag_chain.chain(user)
                print(response.content)
                print('-'*50)

