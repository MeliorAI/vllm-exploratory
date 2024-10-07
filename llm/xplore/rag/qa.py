from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.types import Collection
from chromadb.utils import embedding_functions
from funcy import chunks
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.openai_functions.base import create_structured_output_chain

# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
from openai import OpenAIError
from rich import print as rprint
from tqdm import tqdm

from llm.xplore import DEFAULT_MODEL, get_logger
from llm.xplore.rag.exceptions import NoDocumentVectorError, RetrieverNotInitializedError
from llm.xplore.rag.schemas import (
    QuestionType,
    question_formats,
    question_type_descriptions,
)

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 3500
DEFAULT_CHUNK_OVERLAP = 500

DEFAULT_CHROMA_URI = "localhost:7000"
DEFAULT_CHROMA_COLLECTION_NAME = "llm.xplore"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = DEFAULT_MODEL


Q_CLASSIFICATION_PROMPT = (
    "Determine the question type for the following question: {question}\n\n. "
    "Return only the question type from the provided options.\n"
    "Options:\n{options_prompt}"
)


def printing_qa(query: str, answer: str) -> None:
    """Print the question and answer in a formatted way"""
    # Print the result
    rprint("\n\n> ❓️ [bold]Question:[/bold]")
    rprint(query)
    rprint("\n\n> 🤓 [bold]Answer:[/bold]")
    rprint(answer)


def printing_sources(docs: list[Document]) -> None:
    """Print the sources (documents) used for the answer"""
    # Print the relevant sources used for the answer
    for i, document in enumerate(docs):
        rprint(f"\n> 📚️ [bold]'SOURCE {i}: {document}':[/bold]")


class SentenceTransformerEmbeddingFunctionFix(
    embedding_functions.SentenceTransformerEmbeddingFunction
):
    """ChromaDB & Langchain are a piece of 💩 when it comes to interfaces & documentation.
    Apparently SentenceTransformerEmbeddingFunction misses the following functions:
     - '.embed_documents()'
     - '.embed_query()'

    """
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    def embed_documents(self, input: Documents) -> Embeddings:
        return self.__call__(input)

    def embed_query(self, query:str):
        v = self.__call__(query)
        return v[0].tolist()


class DocQAClient:
    def __init__(
        self,
        openai_api_url: str,
        openai_api_key: str = "EMPTY",
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chroma_uri: str = DEFAULT_CHROMA_URI,
        chroma_collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        # ✂️ Text chunking configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 🎛️ Retrieval configuration (hard-coded for now)
        self.target_source_chunks = 4
        self.chain_type = "stuff"

        # 🤖 OpenAI configuration
        logger.info("🧬 Loading Embedding model")
        self.embeddings = SentenceTransformerEmbeddingFunctionFix(
            model_name=DEFAULT_EMBEDDING_MODEL
        )
        self.llm = ChatOpenAI(
            model=llm_model,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_url,
            temperature=0,
        )  # type: ignore[call-arg]

        # 🌈 ChromaDB
        self._init_db(chroma_uri, chroma_collection_name)

        # QA retriever
        self._init_retriever()

    def _init_db(self, chroma_uri, collection_name):
        logger.info("🛍️ Initializing vectorstore")
        self.chroma_uri = chroma_uri
        self.collection_name = collection_name

        host, port = chroma_uri.split(":")

        self.db_client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )
        self.db = Chroma(
            client=self.db_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        self._init_collection()

    def _init_collection(self) -> Collection:
        try:
            col = self.db_client.get_collection(self.collection_name)
            logger.info(f"🥳 Collection '{self.collection_name}' found!")
            logger.debug(f"Embedding function: {col._embedding_function}")
            logger.debug(f"{col} | entries={col.count()}")
        except Exception as e:
            logger.warning(f"⚠️ Collection not found. {e}")
            col = self.db_client.create_collection(
                self.collection_name, embedding_function=self.embeddings
            )

        return col

    def _init_retriever(self) -> dict:
        self.doc_prompt = PromptTemplate(
            template="Content: {page_content}\n:",
            input_variables=["page_content"],
        )
        # It will init all types of retrievers
        logger.info(f"⛓️  Initializing QA chain (chain={self.chain_type})")
        retriever = self.db.as_retriever(
            search_kwargs={"k": self.target_source_chunks},
            verbose=True,
            return_source_documents=True,
        )
        self.qa = {}
        for question_type, question_value in question_formats.items():
            chain_prompt, prompt_text = self._get_prompt_messages(
                question_value["message"]
            )
            chain = create_structured_output_chain(
                output_schema=question_value["schema"],
                llm=self.llm,
                prompt=chain_prompt,
            )
            final_qa_chain_pydantic = StuffDocumentsChain(
                llm_chain=chain,
                document_variable_name="context",
                document_prompt=self.doc_prompt,
            )
            self.qa[question_type] = {
                "retriever": RetrievalQA(
                    combine_documents_chain=final_qa_chain_pydantic,
                    retriever=retriever,
                    return_source_documents=True,
                ),
                "prompt_text": prompt_text,
            }

        return self.qa

    @staticmethod
    def _get_prompt_messages(my_message: str) -> tuple[ChatPromptTemplate, str]:
        prompt_messages = [
            SystemMessage(content=("Use structured format.")),
            HumanMessage(content="Answer the question using the following context"),
            HumanMessagePromptTemplate.from_template("{context}"),
            HumanMessagePromptTemplate.from_template("Question: {question}"),
            HumanMessage(content=my_message),
        ]
        prompt_text = "\n".join(
            [i.content for i in prompt_messages if hasattr(i, "content")]
        )
        chain_prompt = ChatPromptTemplate(messages=prompt_messages, input_variables=[])  # type: ignore
        return chain_prompt, prompt_text

    def _collect_retrieval_parameters(self) -> dict:
        current_datetime = datetime.now(tz=timezone.utc)
        return {
            "k": self.target_source_chunks,
            "chain_type": self.chain_type,
            # "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            # "LLM_MODEL": self.LLM_MODEL,
            "time_stamp": current_datetime,
        }

    def _insert_doc_in_db(
        self, doc_chunks: list[Document], batch_size: int = 2
    ) -> list[str]:
        inserted = []
        # NOTE: Insert in batches, otherwise we can get a disconnect from the server
        for batch in tqdm(
            chunks(batch_size, doc_chunks), total=len(doc_chunks) // batch_size
        ):
            ids = self.db.add_documents(batch, embedding_function=self.embeddings)
            inserted.extend(ids)

        logger.info(f"✅ Total inserted: {len(inserted)}")

        return inserted

    def list_documents(self):
        col = self.db_client.get_collection(self.collection_name)
        return {metadata["id"] for metadata in col.get()["metadatas"]}

    def delete_documents(
        self, doc_ids: list[str] | None = None, filters: dict[str, Any] | None = None
    ):
        logger.info(
            f"Deleting documents from collection '{self.collection_name}'",
            logtype="auditlogs",
        )
        if filters is None:
            filters = {}
        if doc_ids is None:
            doc_ids = []
        col = self.db_client.get_collection(self.collection_name)
        col.delete(ids=doc_ids, where=filters)

    def index_document(
        self, doc_pages: list[str], doc_meta: dict[str, Any]
    ) -> list[str]:
        logger.info(f"📚️ Adding {len(doc_pages)} document pages")
        documents = [
            Document(page_content=page, metadata={"page": i, **doc_meta})
            for i, page in enumerate(doc_pages)
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        doc_chunks = text_splitter.split_documents(documents)
        logger.info(
            f"🔪 Split into {len(doc_chunks)} chunks of text "
            f"(max. {self.chunk_size} tokens each)"
        )

        return self._insert_doc_in_db(doc_chunks)

    def get_answers(
        self,
        queries: list[str],
        query_types: list[str],
        filters: dict[str, Any] | None = None,
        print_qa: bool = True,
        print_sources: bool = False,
        status_pub: Callable | None = None,
    ) -> tuple[list[str], list[str], list[list[Document]], dict[str, str]]:
        """This functions gets a list of queries and query types and calls openai to get the answer.
        The question types are to validate that the returned answers are of correct type.
        Helps making the answers more accurate and more tailored to what the user wants.

        Args:
            queries: list of queries
            query_types: list of query types
            filters: dictionary of filters to be used in the retriever-currently document ids
        """

        if filters is None:
            filters = {"id": []}

        if self.db is None:
            raise NoDocumentVectorError(
                "No document vector database found. Has any document been indexed yet?"
            )

        if self.qa is None:
            raise RetrieverNotInitializedError("QA Retriever has not been initialized.")

        questions: list[str] = []
        with get_openai_callback() as cb:
            doc_id = filters["id"]
            filt_doc = {"id": doc_id}
            logger.info(f"\nDocument ID: {doc_id} is being processed\n")
            questions, answers, evidences = [], [], []
            for _, (query_type, query) in enumerate(
                zip(query_types, queries, strict=False)
            ):
                # get the questions
                question = self.qa[query_type]["prompt_text"] + "\n" + query

                # Get the answer from the chain
                try:
                    # add the filters here after retriever has already been initialized
                    self.qa[query_type]["retriever"].retriever.search_kwargs[
                        "filter"
                    ] = filt_doc
                    res = self.qa[query_type]["retriever"].invoke(question)
                    logger.info(
                        f"Query: '{query}' for document {doc_id} "
                        f"got answer {res['result']}"
                    )
                except OpenAIError as openai_error:
                    res = {"result": "", "source_documents": []}
                    error_message = f"OpenAI API Error: {openai_error.__str__}"
                    logger.exception(
                        f"Could not extract answer for query: '{query}' "
                        f"and query type: '{query_type}' for document {doc_id}. "
                        f"Original error: {error_message}",
                        exc_info=True,
                    )

                answer, docs = res["result"], res["source_documents"]
                questions.append(question)
                answers.append(answer)
                evidences.append(docs)

                if print_qa:
                    printing_qa(question, answer)

                    if print_sources:
                        printing_sources(docs)

                # print openAI consumption
                rprint(f"[dim]\n-----\n{cb}\n------\n[dim]")

        return (
            questions,
            answers,
            evidences,
            self._collect_retrieval_parameters(),
        )

    def predict_question_type(self, question: str) -> str:
        """Uses openAI to classify the given question among a set of
        prefefined categories.

        Args:
            question (str): The question to classify

        Raises:
            ValueError: If received an unexpected prediction response

        Returns:
            str: The QuestionType classification
        """
        chat = self.llm  # we will use the same parameters we currently use for openai

        # Construct the prompt based on the descriptions
        options_prompt = "\n".join(
            [f"- {qt.value}: {desc}" for qt, desc in question_type_descriptions.items()]
        )

        prompt = Q_CLASSIFICATION_PROMPT.format(
            question=question, options_prompt=options_prompt
        )

        messages = [HumanMessage(content=prompt)]
        with get_openai_callback() as cb:
            response = chat(messages)  # type: ignore[arg-type]
            # print openAI consumption
            rprint(f"[dim]\n-----\n{cb}\n------\n[dim]")

        # Ensure response.content is treated as a string, regardless of its original type
        answer = str(response.content)
        answer = answer.strip().lower()
        valid_responses = [qt.value.lower() for qt in QuestionType]

        for result in valid_responses:
            if result in answer:
                return result

        raise ValueError(f"Unexpected model response: {answer}")
