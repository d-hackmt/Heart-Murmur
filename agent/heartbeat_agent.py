import os
import json
from typing import Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


# --- Retriever setup ---
def build_retriever(json_path: str):
    """Load JSON, split into chunks, and return a retriever."""
    with open(json_path, "r") as f:
        report_data = json.load(f)

    # Convert JSON to pretty string
    report_text = json.dumps(report_data, indent=2)

    # Wrap into LangChain Document
    docs = [Document(page_content=report_text, metadata={"source": "heartbeat_report"})]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    doc_chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    # Vectorstore + retriever
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore.as_retriever()


# --- Tool wrapper ---
def make_retriever_tool(retriever):
    def heart_retriever_tool(query: str, top_k: Optional[int] = 4) -> list[Document]:
        """
        Retrieve information from heartbeat_report.json.
        Call this tool whenever a user asks about heartbeat analysis results.
        """
        return retriever.get_relevant_documents(query)[:top_k]

    return heart_retriever_tool


# --- Agent factory ---
def build_heartbeat_agent(json_path: str):
    retriever = build_retriever(json_path)
    heart_retriever_tool = make_retriever_tool(retriever)

    ollama_model_client = OllamaChatCompletionClient(model="llama3.2")

    agent = AssistantAgent(
        name="HeartbeatAnalysisAgent",
        model_client=ollama_model_client,
        tools=[heart_retriever_tool],
        description="An agent that answers questions about heartbeat audio analysis results based on classification and signal processing metrics.",
        system_message="""
You are HeartbeatAnalysisAgent, an expert assistant for analyzing heartbeat recordings. 
You have access to pre-computed JSON reports that contain the following metrics:
- Classification results (Artifact, Murmur, Normal)
- BPM (beats per minute) and rhythm regularity
- HRV (Heart Rate Variability) metrics: mean BPM, SDNN, RMSSD, CV
- SNR (signal-to-noise ratio)
- Energy distribution (how signal power is distributed across frequencies)
- S1/S2 amplitude ratio
- Extra peak detection (murmurs, gallops, abnormal beats)
- Irregular spacing statistics
- Frequency band energy (150–500 Hz band for murmurs)
- Visual outputs (waveform, spectrogram, histograms)

Your role:
- Answer user questions using ONLY the data provided in the JSON report or general definitions of the above metrics.
- If asked about medical interpretation, provide general information only (e.g., what a high HRV usually means), not medical advice.
- If a query is unrelated to heartbeat analysis, classification, or signal processing, reply with: "I don't know."
- Do not make up or hallucinate answers.
- And If user greets you with "Hi" or something like "who are you", greet back and reply as well about you.
- Keep explanations clear and concise. 
- When a user explicitly asks you to stop or when a conversation session is complete, respond with "TERMINATE".
""",
        reflect_on_tool_use=True
    )

    return agent
