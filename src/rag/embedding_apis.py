# from sklearn.feature_extraction.text import TfidfVectorizer
#
# class OpenAIEmbedding:
#     def __init__(self):
#         self.vectorizer = None
#         self.texts = []          # 所有文档的文本
#         self.is_fitted = False
#
#     def fit(self, texts):
#         """用全部文档文本训练 TF-IDF 模型"""
#         if not texts:
#             return
#         self.texts = texts
#         self.vectorizer = TfidfVectorizer()   # 每次 fit 都新建一个干净的模型
#         self.vectorizer.fit(texts)
#         self.is_fitted = True
#
#     def embed_documents(self, texts):
#         """批量生成文档向量（训练+转换）"""
#         self.fit(texts)
#         return self.vectorizer.transform(texts).toarray().tolist()
#
#     def embed_query(self, text):
#         """查询文本向量化（必须已训练）"""
#         if not self.is_fitted:
#             # 极端情况兜底：返回 100 维全零（但正常情况不会触发）
#             return [0.0] * 100
#         return self.vectorizer.transform([text]).toarray()[0].tolist()

from openai import OpenAI

class OpenAIEmbedding:
    def __init__(self, model="nomic-embed-text", batch_size=16):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(
            api_key="ollama",  # 任意非空即可
            base_url="http://localhost:11434/v1"
        )

    def embed_documents(self, texts):
        result = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            res = self.client.embeddings.create(input=batch, model=self.model)
            result.extend([d.embedding for d in res.data])
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]