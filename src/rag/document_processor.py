import re
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from typing import List
#from langchain.docstore.document import Document
from langchain_core.documents import Document

class PaperTextSplitter(TextSplitter):
    def __init__(self, chunk_size=500, chunk_overlap=50, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # A simplified approach to split by sections often found in papers
        # This regex looks for patterns that are likely to be section titles.
        # e.g., "1. Introduction", "Abstract", "References"
        sections = re.split(
            r"\n(?=Abstract|Introduction|Conclusion|References|Discussion|Results|Methods|Background|\d+\.\s[A-Z])",
            text,
        )

        # Further split sections if they are too large
        chunks = []
        for section in sections:
            if len(section) > self.chunk_size:
                # If a section is larger than chunk_size, use a simpler splitter for it
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                chunks.extend(recursive_splitter.split_text(section))
            elif section.strip():
                chunks.append(section)
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        new_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["section"] = i + 1
                new_doc = Document(page_content=chunk, metadata=metadata)
                new_docs.append(new_doc)
        return new_docs


class ChapterTitleSplitter(TextSplitter):
    def __init__(self, chunk_size=500, chunk_overlap=50, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # Regex to split by chapters (e.g., "第1章") and titles (e.g., "1.1", "1.1.1")
        # This regex looks for lines starting with chapter/section markers.
        sections = re.split(
            r"\n(?=^第[一二三四五六七八九十\d]+章\s.*|^\d+(?:\.\d+)*\s.*)",
            text,
            flags=re.MULTILINE,
        )

        # Further split sections if they are too large
        chunks = []
        for section in sections:
            if len(section) > self.chunk_size:
                # If a section is larger than chunk_size, use a simpler splitter for it
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                chunks.extend(recursive_splitter.split_text(section))
            elif section.strip():
                chunks.append(section)
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        new_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["section"] = i + 1
                new_doc = Document(page_content=chunk, metadata=metadata)
                new_docs.append(new_doc)
        return new_docs


class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50, strategy="default"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if strategy == "paper":
            self.text_splitter = PaperTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif strategy == "chapter":
            self.text_splitter = ChapterTitleSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        self.loaders = {
            "pdf": PyMuPDFLoader,
            "md": UnstructuredMarkdownLoader,
        }

    def load_documents(self, file_paths):
        """加载多种格式的文档"""
        documents = []
        for file_path in file_paths:
            file_extension = file_path.split(".")[-1]  #获取文件后缀（如 pdf、txt、docx）
            loader_class = self.loaders.get(file_extension)
            # PDF 文件 → 逐页读取文本 → 每页生成 1 个 Document → 返回 Document 列表
            if loader_class:
                loader = loader_class(file_path)
                documents.extend(loader.load())
        return documents

    # def clean_text(self, text):
    #     """清洗文本数据"""
    #     # 移除中日韩字符之间的换行符
    #     text = re.sub(r"([^\u4e00-\u9fa5\n])\n([^\u4e00-\u9fa5\n])", r"\1 \2", text)
    #     # 移除特殊符号和多余的空格
    #     text = text.replace("•", "").replace(" ", "").replace("\n\n", "\n")
    #     return text
    def clean_text(self, text):
        # 1. 移除图片占位符（形如 image[[...]]）
        text = re.sub(r'image\[\[.*?\]\]', '', text)
        # 2. 移除无法打印的控制字符（保留换行）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # 3. 只保留中英文、数字、常见标点、空格和换行
        text = re.sub(
            r'[^\u4e00-\u9fff\uff00-\uffefa-zA-Z0-9，。；：？！“”‘’（）【】《》—… ,.;:?!()\[\]{}<>\-+=*/\\@#$%^&_|~`\n\t]', ' ',
            text)
        # 4. 合并连续空格，保留单个换行
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' *\n *', '\n', text)
        return text.strip()

    def process_documents(self, file_paths):
        docs = self.load_documents(file_paths)
        for doc in docs:
            doc.page_content = self.clean_text(doc.page_content)
        # 过滤掉内容过短或全为符号的块
        docs = [doc for doc in docs if
                len(doc.page_content.strip()) > 10 and re.search(r'[\u4e00-\u9fff]', doc.page_content)]  # 至少有一个中文字
        split_docs = self.text_splitter.split_documents(docs)
        return split_docs