# import the necessary libraries
from llama_index.core import (
    SimpleDirectoryReader, # for loading the data
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore # for vector store
from IPython.display import Markdown, display
from llama_index.embeddings.ollama import OllamaEmbedding
import faiss
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response.notebook_utils import display_source_node
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from toxic_rail import Toxic_Rail
from llama_index.core import Settings

### The prompt formats for llama3 are taken from this - https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

class Helper:
    # initialize the helper class
    def __init__(self):
        self.llm = ChatOllama(model="llama3", format="json", temperature=0)
        self.embed_model = OllamaEmbedding(model_name="mxbai-embed-large:latest")
        Settings.embed_model = self.embed_model

        # define the grader prompt
        self.grader_prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved context: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
            input_variables=["question", "context"],
        )
        self.retrieval_grader = self.grader_prompt | self.llm | JsonOutputParser()
        
        self.generate_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
                Question: {question} 
                Context: {context} 
                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "context"],
        )
        #
        self.generator = self.generate_prompt | self.llm | StrOutputParser()
        self.retriever = None
        
        self.hallucination_grader_prompt = PromptTemplate(
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts: {context} 
            Here is the answer: {answer}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["context", "answer"],
        )
        self.hallucination_grader = self.hallucination_grader_prompt | self.llm | JsonOutputParser()
        self.guardail = Toxic_Rail(mode="predict")

#   
    def load_index(self, path):
        print("---LOADING INDEX FROM PERSISTENNT STORE---")        
        vector_store = FaissVectorStore.from_persist_path(path + "/default__vector_store.json")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=path)
        retrieved_index = load_index_from_storage(storage_context=storage_context)
        self.retriever = retrieved_index.as_retriever()
        return self.retriever
    
    # retrieve the context
    def retrieve_context(self, state):
        print("---RETRIEVE---")
        question = state["question"]
        context = self.retriever.retrieve(question)
        return {"context": context, "question": question}
    
    # grade the chunks
    def grade_chunks(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        context = state["context"]

        # Score each chunk
        filtered_docs = []
        for d in context:
            score = self.retrieval_grader.invoke(
                {"question": question, "context": d.text})
            grade = score["score"]
            print(d.text)
            print(score)
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"context": filtered_docs, "question": question}

    # decision to generate or stop
    def generation_decision(self, state):
        context = state["context"]
        if(len(context) == 0):
            return "stop"
        else:
            return "generate"
    
    def guardrail_decision(self, state):
        classification = state["question_classification"]
        if "|toxic|" in classification:
            return "stop"
        else:
            return "retrieve_context"
        
    def generate(self, state):
        print("---GENERATE---")
        question = state["question"]
        context = state["context"]

        # RAG generation
        answer = self.generator.invoke({"context": context, "question": question})
        print(f"Generated. Question: {question}, Answer: {answer}")
        return {"context": context[0].text, "question": question, "answer": answer}

    def grade_hallucination(self, state):
        print("---CHECK HALLUCINATIONS---")
        context = state["context"]
        answer = state["answer"]
        question = state["question"]
        score = self.hallucination_grader.invoke({"context": context, "answer": answer})
        print(score)    
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            return {"quality": "good", "answer": answer, "context": context, "question": question}
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return {"quality": "not good", "answer": "No relevant answer available in the knowledgebase",
                   "context": context, "question": question}

    def guardtail_check(self, state):
        print("---CHECK FOR TOXICITY---")
        question = state["question"]
        classification = self.guardail.predict(question)[0]
        if "|toxic|" in classification:
            print("---CLASSIFICASTION is TOXIC--")
        else:
            print("---CLASSIFICASTION is NON_TOXIC--")
        state["question_classification"] = classification
        return state
