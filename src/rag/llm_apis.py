# import os
# from openai import OpenAI
# from dotenv import load_dotenv, find_dotenv
#
#
# class LLMClient:
#     def __init__(self):
#         load_dotenv(find_dotenv())
#         # api_key = os.getenv("sk-3e79a72561ff4788ba9c687ea9a6b620")
#         # base_url = os.getenv("https://dashscope.aliyuncs.com/compatible-mode/v1")
#         base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#         api_key = "sk-3e79a72561ff4788ba9c687ea9a6b620"
#         self.model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-8B")
#
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
#
#         self.client = OpenAI(api_key=api_key, base_url=base_url)
#
#     def generate_answer(self, question, context):
#         """
#         Generates an answer using the LLM based on the provided question and context.
#         """
#         context_str = "\n\n".join(context)
#         prompt = f"请根据以下提供的知识回答问题：\n\n{context_str}\n\n问题：{question}"
#         print(f"LLM Input: {prompt}")
#
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "你是一个问答机器人，请根据提供的背景知识回答问题。",
#                 },
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0.7,
#
#         # 0.0 → 严谨、事实准确、不发散
#         # 1.0 → 创意高、可能编内容
#         # 0.7
#         # 适合问答场景
#         )
#         return response.choices[0].message.content

# 纯本地 LLM 模拟，不联网、不报错、直接出答案
# class LLMClient:
#     def __init__(self, model="local-llm"):
#         self.model = model
#
#     def generate_answer(self, question, context):
#         # 直接本地返回答案，不调用任何 API
#         return "操作系统是管理计算机硬件与软件资源的计算机程序，同时也是计算机系统的内核与基石。"

from openai import OpenAI

class LLMClient:
    def __init__(self, model="deepseek-r1:8b"):
        # 关键：连接本地 Ollama
        self.client = OpenAI(
            api_key="ollama",  # 随便填，不需要真实key
            base_url="http://localhost:11434/v1"  # 本地 Ollama 地址
        )
        self.model = model  # 你在ollama里用的模型名

    def generate_answer(self, question, context):
        prompt = f"请根据以下知识回答问题：\n{context}\n问题：{question}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content