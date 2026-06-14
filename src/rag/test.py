import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1. 加载PDF并切分
loader = PyMuPDFLoader("D:\\408agent\\408-RAG-main\\408-RAG-main\\data_base\\knowlege_db\\1.1_1_操作系统的概念、功能.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 2. 向量化存储（用Ollama的embedding，比如 nomic-embed-text）
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

# 3. 对接本地 deepseek-r1:8b
llm = Ollama(model="deepseek-r1:8b", temperature=0)

# 4. 构建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 5. 问答
question = "操作系统向上层用户提供了哪几种服务接口？"
print(qa_chain.run(question))