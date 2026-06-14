import json, os
import numpy as np
from langchain_core.documents import Document
from embedding_apis import OpenAIEmbedding
import jieba

class VectorDatabase:
    def __init__(self, persist_directory=None):
        self.persist_directory = persist_directory
        self.documents = []
        self.embeddings = []
        self.embedding = OpenAIEmbedding()

    def create_from_documents(self, documents, persist_directory=None):
        if persist_directory:
            self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # 过滤并保存文档
        texts = [doc.page_content for doc in documents if doc.page_content.strip()]
        self.documents = [doc for doc in documents if doc.page_content.strip()]

        # 训练并生成向量
        self.embeddings = self.embedding.embed_documents(texts)

        # 保存到磁盘（纯文本 + 向量数字）
        save_path = os.path.join(self.persist_directory, "vector_db.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "embeddings": self.embeddings,
                "docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in self.documents]
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 向量库构建完成：{len(self.embeddings)} 条")
        return self

    def load_existing(self, persist_directory):
        self.persist_directory = persist_directory
        with open(os.path.join(persist_directory, "vector_db.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data["docs"]]
        self.embeddings = data["embeddings"]
        # 不再需要 fit，因为嵌入模型通过 API 直接生成向量
        return self

    def similarity_search(self, query, k=3, score_threshold=0.15, alpha=0.15):
        """
        混合相似度检索：
        alpha: 向量相似度的权重 (0~1)，关键词相似度权重为 1-alpha
        """
        # 1. 向量相似度
        q_vec = np.array(self.embedding.embed_query(query))
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            print("警告：查询向量全零")
            return []

        vec_scores = []
        for vec in self.embeddings:
            vec = np.array(vec)
            d_norm = np.linalg.norm(vec)
            score = np.dot(q_vec, vec) / (q_norm * d_norm) if d_norm != 0 else 0.0
            vec_scores.append(score)

        # 2. 关键词匹配分数（Jaccard相似度）
        query_tokens = set(jieba.lcut(query.lower()))
        kw_scores = []
        for doc in self.documents:
            doc_tokens = set(jieba.lcut(doc.page_content.lower()))
            if not query_tokens or not doc_tokens:
                kw_scores.append(0.0)
            else:
                intersection = query_tokens & doc_tokens
                union = query_tokens | doc_tokens
                kw_scores.append(len(intersection) / len(union))

        # 3. 融合分数
        final_scores = [
            alpha * vec_scores[i] + (1 - alpha) * kw_scores[i]
            for i in range(len(vec_scores))
        ]

        # 4. 排序并过滤
        sorted_idx = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        top_idx = [i for i in sorted_idx if final_scores[i] >= score_threshold][:k]

        print(f"检索到 {len(top_idx)} 个相关文档（混合阈值 {score_threshold}）")
        for idx in top_idx:
            snippet = self.documents[idx].page_content[:60].replace('\n', ' ')
            print(
                f"混合分数: {final_scores[idx]:.4f} | 向量: {vec_scores[idx]:.4f} 关键词: {kw_scores[idx]:.4f} | 片段: {snippet}...")
        return [self.documents[i] for i in top_idx]

        # print(f"检索到 {len(top_idx)} 个相关文档（混合阈值 {score_threshold}）")
        # for i in top_idx:
        #     snippet = self.documents[i].page_content[:60].replace('\n', ' ')
        #     print(f"混合分数: {final_scores[i]:.4f} | ...")
        #
        # top_docs = [self.documents[i] for i in top_idx]
        # top_scores = [final_scores[i] for i in top_idx]
        # return top_docs, top_scores

    def get_collection_count(self):
        return len(self.documents)