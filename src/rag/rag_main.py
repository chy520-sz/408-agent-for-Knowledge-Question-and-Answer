import os
import logging
from dotenv import load_dotenv, find_dotenv
from document_processor import DocumentProcessor
from vector_db import VectorDatabase
from llm_apis import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv(find_dotenv())

class RAGSystem:
    def __init__(self, persist_dir, strategy="default"):
        self.strategy = strategy
        self.persist_dir = persist_dir
        self.document_processor = DocumentProcessor(strategy=strategy)
        self.llm_client = LLMClient()

    def build_knowledge_base(self, data_dir):
        """构建知识库（仅在首次运行时执行）"""
        # 如果已经存在向量库文件，跳过
        if os.path.exists(os.path.join(self.persist_dir, "vector_db.json")):
            logging.info("知识库已存在，跳过构建")
            return

        # 收集所有文件
        file_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_paths.append(os.path.join(root, file))

        # 处理文档
        processed_docs = self.document_processor.process_documents(file_paths)

        # 保存切割后的文档（可选，调试用）
        output_dir = os.path.join(os.path.dirname(self.persist_dir), "output", self.strategy)
        os.makedirs(output_dir, exist_ok=True)
        for doc in processed_docs:
            source_name = os.path.basename(doc.metadata.get("source", "unknown"))
            base_name = os.path.splitext(source_name)[0].replace(" ", "_")
            doc_dir = os.path.join(output_dir, base_name)
            os.makedirs(doc_dir, exist_ok=True)
            chunk_num = 1
            while os.path.exists(os.path.join(doc_dir, f"chunk_{chunk_num}.txt")):
                chunk_num += 1
            with open(os.path.join(doc_dir, f"chunk_{chunk_num}.txt"), "w", encoding="utf-8") as f:
                f.write(doc.page_content)

        # 构建向量库（这里会保存嵌入和向量化器）
        vec_db = VectorDatabase(persist_directory=self.persist_dir)
        vec_db.create_from_documents(processed_docs, persist_directory=self.persist_dir)
        logging.info(f"知识库构建完成，共 {vec_db.get_collection_count()} 块")

    def query(self, question, k=15):
        """查询：每次独立加载向量库，确保向量化器正确恢复"""
        # 每次都新建一个 VectorDatabase 并从磁盘加载（避免实例状态污染）
        vec_db = VectorDatabase()
        vec_db.load_existing(self.persist_dir)
        docs = vec_db.similarity_search(question, k=k, score_threshold=0.15)

        if not docs:
            return "未找到与问题相关的知识库内容。"

        context = [d.page_content for d in docs]
        return self.llm_client.generate_answer(question, context)

    # def query(self, question, k=5, threshold=0.15):
    #     """查询并返回答案与参考文档"""
    #     vec_db = VectorDatabase()
    #     vec_db.load_existing(self.persist_dir)  # 每次独立加载，避免状态污染
    #     docs, scores = vec_db.similarity_search(question, k=k, score_threshold=threshold)
    #
    #     if not docs:
    #         return "未找到与问题相关的知识库内容。", []
    #
    #     context = [d.page_content for d in docs]
    #     answer = self.llm_client.generate_answer(question, context)
    #
    #     # 构建带分数的参考片段
    #     source_docs = [
    #         {"content": doc.page_content, "score": score}
    #         for doc, score in zip(docs, scores)
    #     ]
    #     return answer, source_docs

if __name__ == "__main__":
    # 项目根目录（408-RAG-main）
    current_file = os.path.abspath(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

    knowledge_base_dir = os.path.join(project_dir, "data_base", "knowlege_db")
    persist_directory = os.path.join(project_dir, "data_base", "vector_db", "408.db")
    os.makedirs(persist_directory, exist_ok=True)

    print("项目根目录：", project_dir)
    print("知识库路径：", knowledge_base_dir)

    rag = RAGSystem(persist_dir=persist_directory, strategy="chapter")

    # 构建知识库（如果不存在）
    logging.info("开始构建知识库...")
    rag.build_knowledge_base(data_dir=knowledge_base_dir)
    logging.info("知识库构建完成。")

    # 查询测试
    logging.info("执行查询...")
    answer = rag.query("什么是操作系统")
    logging.info(f"最终答案: {answer}")