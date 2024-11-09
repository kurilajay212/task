import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load a dataset (using a sample dataset for this demo)
dataset = load_dataset("ag_news", split="test[:10%]")  # using a smaller subset for demonstration

# Initialize models
retriever = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence Transformer for retrieval
generator = pipeline("text-generation", model="distilgpt2")  # GPT-2 model for generation

# Define the RAG Evaluation function
def rag_evaluation(query, documents):
    # Step 1: Retrieve relevant documents
    query_embedding = retriever.encode(query)
    doc_embeddings = retriever.encode([doc["text"] for doc in documents])
    scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    
    # Get the highest scoring document
    top_doc_index = scores.argmax().item()
    top_document = documents[top_doc_index]["text"]
    logger.info(f"Top retrieved document: {top_document} with score: {scores[top_doc_index].item()}")

    # Step 2: Generate answer based on the retrieved document
    input_text = f"Context: {top_document}\nQuestion: {query}"
    generated_text = generator(input_text, max_new_tokens=50, num_return_sequences=1)[0]["generated_text"]
    logger.info(f"Generated Answer: {generated_text}")

    # Return the result and score for evaluation purposes
    return {"query": query, "retrieved_document": top_document, "generated_answer": generated_text, "retrieval_score": scores[top_doc_index].item()}

# Sample Query
# query = "What are the latest advancements in artificial intelligence?"
query2="What organization is responsible for announcing Sudan's troop withdrawal from Darfur?"
# Perform RAG Evaluation
results = rag_evaluation(query2, dataset)

# Output Results
logger.info(f"Final Evaluation Result: {results}")
