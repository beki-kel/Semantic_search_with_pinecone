from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import spacy
from nltk.corpus import wordnet
from nltk import download
import torch
from typing import List, Dict, Any

download('wordnet')
download('omw-1.4')

def create_pinecone_index(api_key: str, index_name: str, dimension: int, metric: str = "cosine"):
    """
    Create a Pinecone index.
    :param api_key: Pinecone API key.
    :param index_name: Name of the index.
    :param dimension: Dimension of the embeddings.
    :param metric: Distance metric to use (default is 'cosine').
    :return: Pinecone Index object.
    """
    pc = Pinecone(api_key=api_key)
    Indexes_list = pc.list_indexes()
    checkList = [index["name"] for index in Indexes_list]

    if index_name in checkList:
        print(f"Index '{index_name}' already exists.")
    else:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{index_name}' created.")

    return pc.Index(index_name)

class SemanticSearchSystem:
    def __init__(self, dataset_name: str = "quora", dataset_split: str = "train[260000:290000]",
                 embedding_model: str = "all-MiniLM-L6-v2", pinecone_index=None,
                 batch_size: int = 200, vector_limit: int = 30000):
        """
        Initialize the Semantic Search system.
        :param dataset_name: Name of the dataset to load.
        :param dataset_split: Split of the dataset.
        :param embedding_model: Pre-trained Sentence-Transformer model.
        :param pinecone_index: Pre-created Pinecone index instance.
        :param batch_size: Number of embeddings to process in each batch.
        :param vector_limit: Limit on number of questions to process.
        """
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.vector_limit = vector_limit
        self.model = SentenceTransformer(self.embedding_model, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Load Dataset
        self.dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        self.questions = [q['text'] for q in self.dataset['questions']]
        self.questions = self.questions[:self.vector_limit]
        print(f"Loaded {len(self.questions)} questions.")

        # Assign the Pinecone index
        self.index = pinecone_index
        if not self.index:
            raise ValueError("Pinecone index must be provided.")

        # Load SpaCy for NLP processing
        self.nlp = spacy.load("en_core_web_sm")
        self._upsert_all_embeddings()

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query text using SpaCy for lemmatization and stop-word removal.
        :param query: The raw query text.
        :return: Preprocessed query string.
        """
        doc = self.nlp(query)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)

    def expand_query(self, query: str) -> List[str]:
        """
        Query Expansion using WordNet synonyms.
        :param query: Preprocessed query.
        :return: List of expanded query terms.
        """
        synonyms = set()
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
        return list(synonyms)
    def _upsert_all_embeddings(self):
      """
      Upsert all embeddings to Pinecone index in batches.
      """
      print("Starting to upsert embeddings to Pinecone...")
      for start in range(0, len(self.questions), self.batch_size):
          end = min(start + self.batch_size, len(self.questions))
          batch = self.questions[start:end]
          embeddings = self.create_embeddings(batch)
          self.upsert_embeddings_to_pinecone(embeddings, batch, start)
      print("All embeddings upserted successfully.")

    def create_embeddings(self, batch: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of queries.
        :param batch: List of queries to process.
        :return: List of embeddings for the batch.
        """
        return self.model.encode(batch, convert_to_numpy=True)

    def upsert_embeddings_to_pinecone(self, embeddings: List[List[float]], batch: List[str], start_index: int):
        """
        Upload embeddings to Pinecone.
        :param embeddings: List of embeddings to upload.
        :param batch: List of original questions corresponding to the embeddings.
        :param start_index: Starting index for the batch in the dataset.
        """
        ids = [str(start_index + i) for i in range(len(embeddings))]
        metadatas = [{'text': text} for text in batch]
        records = zip(ids, embeddings, metadatas)
        self.index.upsert(vectors=list(records))

    def run_query(self, query: str, top_k: int = 5):
        """
        Run a query against the Pinecone index and retrieve the top-k most similar questions.
        :param query: User input query.
        :param top_k: Number of top results to retrieve.
        :return: List of top-k results.
        """
        preprocessed_query = self.preprocess_query(query)
        expanded_query = self.expand_query(preprocessed_query)
        combined_query = " ".join(expanded_query) or preprocessed_query
        query_embedding = self.model.encode(combined_query, convert_to_numpy=True)

        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        return results['matches']


    def display_results(self, results: List[Dict[str, Any]]):
        """
        Display search results.
        :param results: Results from Pinecone query.
        """
        for result in results:
            score = round(result['score'], 2)
            text = result['metadata']['text']
            print(f"Score: {score}, Passage: {text}")

    def run_full_pipeline(self, query: str, top_k: int = 5):
        """
        Run the full semantic search pipeline: Query Understanding, Content Enrichment, and Semantic Matching.
        :param query: User input query.
        :param top_k: Number of top results to retrieve.
        """
        print(f"Running query: {query}")

        # Step 1: Preprocess query
        preprocessed_query = self.preprocess_query(query)
        print(f"Preprocessed query: {preprocessed_query}")

        # Step 2: Query Expansion (Content Enrichment)
        expanded_query = self.expand_query(preprocessed_query)
        print(f"Expanded query: {expanded_query}")

        # Step 3: Run Query
        results = self.run_query(expanded_query, top_k)

        # Step 4: Display Results
        self.display_results(results)