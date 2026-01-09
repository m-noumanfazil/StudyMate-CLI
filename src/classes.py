from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
import os

load_dotenv()
# Loads environment variables from .env at import time.
# NOTE: This does NOT validate whether required keys (like GROQ_API_KEY) exist.
# Failure is deferred to LLM initialization later, which can slow debugging.


class vectordb:
    def __init__(self):
        # Embedding model used for ALL sessions and ALL documents.
        # IMPORTANT: If this model name is ever changed later,
        # existing Chroma collections become semantically incompatible.
        self.embedding_model_name = os.environ.get("EMBEDDING_MODEL")

        # Embedding engine instance.
        # This is reused across all collections.
        self.embedding_engine = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )

        # Single persist directory for all Chroma collections.
        # All sessions live inside this folder.
        # Fine for a CLI project, risky for multi-user or multi-project setups.
        self.persist_directory = "./chroma_db"

        # In-memory registry of active sessions:
        # session_name -> Chroma collection object
        # This dictionary is the runtime source of truth.
        self.collections = {}

        # Text splitter configuration.
        # Chunk size and overlap are fixed.
        # No adaptive behavior based on document type or length.
        self.textsplitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )

        # Load previously created sessions from sessions.txt.
        # This assumes sessions.txt is accurate and in sync with Chroma.
        self.load_all_sessions()

        print("Vector database initialized successfully.")


    def load_all_sessions(self):
        # If this is the first run, sessions.txt will not exist.
        # Silent return means no feedback that the DB is empty by design.
        if not os.path.exists("sessions.txt"):
            return

        # Read session names line-by-line.
        # Empty lines are ignored.
        with open("sessions.txt", "r") as f:
            session_names = [line.strip() for line in f if line.strip()]

        # Recreate Chroma collection handles for each saved session.
        # ASSUMPTIONS (not validated):
        # - Collection actually exists in Chroma
        # - Embedding model matches stored vectors
        # - Persist directory is intact
        for name in session_names:
            self.collections[name] = Chroma(
                collection_name=name,
                embedding_function=self.embedding_engine,
                persist_directory=self.persist_directory
            )


    def _save_session_name(self, session_name):
        # Appends session name to sessions.txt.
        # No deduplication.
        # If logic elsewhere fails, duplicate entries are possible.
        with open("sessions.txt", "a") as f:
            f.write(session_name + "\n")


    def _remove_session_name(self, session_name):
        # Removes a session name from sessions.txt.
        # Entire file is rewritten.
        # Works for small scale, not concurrency-safe.
        if not os.path.exists("sessions.txt"):
            return

        with open("sessions.txt", "r") as f:
            names = [line.strip() for line in f if line.strip()]

        names = [name for name in names if name != session_name]

        with open("sessions.txt", "w") as f:
            for name in names:
                f.write(name + "\n")


    def create_session(self, session_name):
        """
        Attempts to create a new session.
        Returns True if session was created successfully.
        Returns False if session already exists.
        """
        if session_name in self.collections:
            return False  # session exists, do not create
        # Create new Chroma collection
        self.collections[session_name] = Chroma(
            collection_name=session_name,
            embedding_function=self.embedding_engine,
            persist_directory=self.persist_directory
        )
        # Persist the session name
        self._save_session_name(session_name)
        return True

    def list_session(self):
        # Lists all session names currently loaded in memory.
        # Order depends on dictionary insertion order.
        # This may not strictly reflect the order in sessions.txt.
        if self.collections is None:
            print("Session List is empty. First add some sessions.")
            return 
        print("List of all Session Names")
        print("-------------------------")
        for i, name in enumerate(self.collections):
            print(f"Session no {i+1}: {name}")
        print("-------------------------")


    def get_session(self, session_name):
        """
        Retrieve the Chroma collection for a given session name.
        Returns the collection object if exists, else None.
        """
        # Caller must handle None safely.
        # This method also prints a warning, which may cause duplicated warnings
        # if caller prints its own error messages.
        if session_name in self.collections:
            return self.collections[session_name]
        else:
            print(f"[WARN] Session '{session_name}' does not exist.")
            return None


    def chunk_document(self, document):
        # Splits a loaded document into chunks using predefined splitter.
        # Thin wrapper function; mainly for readability and future extensibility.
        chunks = self.textsplitter.split_documents(document)
        return chunks


    def add_file(self, documents_list, collection_name):
        # Adds one or more PDF documents to a given collection.
        # documents_list is assumed to contain valid PDF file paths.
        
        # Retrieve target collection from in-memory registry.
        collection = self.collections[collection_name]

        for i, docs_path in enumerate(documents_list):
            all_chunks = []

            # Loads PDF file.
            # No try/except: invalid path or corrupted PDF will crash execution.
            loader = PyPDFLoader(docs_path)
            document = loader.load()

            # Split document into chunks.
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)

            # Add chunks to Chroma.
            # No deduplication logic.
            # Re-adding same document will duplicate embeddings.
            collection.add_documents(all_chunks)

            print(f"Document no {i+1} added successfully.")


    def delete_session(self, session_name):
        # Deletes the entire Chroma collection permanently.
        # No confirmation step.
        # Irreversible operation.
        self.collections[session_name].delete_collection()

        # Remove from sessions.txt persistence.
        self._remove_session_name(session_name)

        # Remove from in-memory registry.
        del self.collections[session_name]

        print(f"Session {session_name} deleted successfully.")


class RAGAssistant:
    def __init__(self, vector_database):
        """
        Initialize RAG Assistant with:
        - Vector database reference
        - Groq LLM
        - Prompt template
        """
        # Initialize LLM immediately.
        # If GROQ_API_KEY is missing, program crashes here.
        self.llm = self._initialize_llm()

        # Store reference to vectordb instance.
        # Tight coupling: RAGAssistant assumes vectordb interface.
        self.vector_db = vector_database

        # System-level prompt defining behavior and strict grounding rules.
        system_msg = SystemMessagePromptTemplate.from_template(
        """You are StudyMate, a helpful educational assistant.
           You are answering questions using materials from the following session: {session_name}.
           Treat this as the topic or collection of documents you can use to answer questions.

           RULES:
           1. Only use the context provided below to answer questions.
           2. If the answer is not in the context, respond exactly:
              "I don't know. No relevant information found."
           3. Do not guess, assume, or add information not present in the context.
           4. Provide concise, structured, and easy-to-understand answers.
           5. Use bullet points for lists if needed, and keep explanations simple.
           6. Context will always be provided separately. Do not answer questions beyond the context.

           INPUT DESCRIPTION:
           - Name of the session/collection: {session_name}
           - Relevant materials/context: {context}
           - User Question:{question}
           """
        )

        # Human message injects the actual question and context.
        # Context is duplicated conceptually between system and human messages.
        human_msg = HumanMessagePromptTemplate.from_template(
            "Question: {question}\nContext: {context}"
        )

        # Final prompt template.
        self.prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

        # Simple linear chain: Prompt -> LLM
        # No memory, no tool usage, no retries at chain level.
        self.chain = self.prompt | self.llm

        print("[INFO] RAGAssistant initialized successfully.")


    def _initialize_llm(self):
        # Fetch Groq API key from environment.
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # Hard failure by design.
            # Acceptable for CLI learning project.
            raise ValueError("Groq API key not found in environment variables!")

        # Groq chat model configuration.
        return ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0,
            reasoning_format="hidden",
            max_retries=2
        )


    def query(self, session_name: str, question: str, n_results: int = 5):
        """
        Retrieve relevant chunks from the vector DB for a session and
        query the LLM.

        Args:
            session_name: Name of the session/collection to query
            question: User's question
            n_results: How many top chunks to retrieve
        """
        # Retrieve collection from vector DB.
        collection = self.vector_db.get_session(session_name)
        if collection is None:
            # Defensive fallback if session does not exist.
            return "I don't know. No relevant information found."

        # Perform similarity search in Chroma.
        # No score threshold — low-relevance chunks may still be returned.
        docs = collection.similarity_search(question, k=n_results)

        if not docs:
            # Explicit handling when no chunks are retrieved.
            return "I don't know. No relevant information found."

        # Combine retrieved chunks into a single context string.
        # Metadata (page number, source file) is discarded.
        context = "\n\n".join(doc.page_content for doc in docs)

        # Invoke LLM with required prompt variables.
        # Prompt-variable mismatch here will raise runtime errors.
        response = self.chain.invoke({
            "question": question,
            "context": context,
            "session_name": session_name
        })

        # Return plain text response.
        return response.content

