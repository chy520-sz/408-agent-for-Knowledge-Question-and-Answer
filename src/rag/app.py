import streamlit as st
import sys
import os

# 将项目src路径加入，确保能导入内部模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'rag'))

from rag_main import RAGSystem  # 复用已有的RAGSystem类
from dotenv import load_dotenv, find_dotenv
import logging

# 加载环境变量（API密钥等）
load_dotenv(find_dotenv())

# 页面配置
st.set_page_config(
    page_title="408考研知识问答",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS美化界面
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatMessage.user {
        background-color: #e3f2fd;
    }
    .stChatMessage.assistant {
        background-color: #f5f5f5;
    }
    .source-box {
        background-color: #fafafa;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    # 配置知识库路径（根据你的实际路径调整）
    project_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_base_dir = os.path.join(project_dir, "data_base", "knowlege_db")
    persist_dir = os.path.join(project_dir, "data_base", "vector_db", "408.db")

    # 确保目录存在
    os.makedirs(os.path.dirname(persist_dir), exist_ok=True)

    with st.spinner("正在初始化知识库，请稍候..."):
        rag = RAGSystem(persist_dir=persist_dir, strategy="chapter")
        # 构建知识库（如果已存在则跳过）
        rag.build_knowledge_base(data_dir=knowledge_base_dir)
        st.session_state.rag_system = rag
        st.success("知识库初始化完成！")

# 侧边栏：检索参数控制
with st.sidebar:
    st.header("⚙️ 检索设置")
    k = st.slider("检索文档数量", min_value=1, max_value=10, value=5)
    threshold = st.slider("相似度阈值", min_value=0.1, max_value=0.9, value=0.15, step=0.05)
    st.markdown("---")
    st.markdown("### 📚 关于本系统")
    st.markdown("""
    基于 **RAG 技术** 的 408 考研知识问答系统。  
    - 知识库来源：王道考研系列
    - 模型：本地 Ollama 模型  
    """)

# 主界面标题
st.title("🎓 408考研知识问答 Agent")
st.caption("基于检索增强生成（RAG）的计算机考研知识答疑助手")

# 显示历史聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果是助手回答且有参考来源，展示可折叠的参考资料
        if msg["role"] == "assistant" and "source_docs" in msg:
            with st.expander("📖 查看参考片段"):
                for i, doc in enumerate(msg["source_docs"], 1):
                    st.markdown(
                        f'<div class="source-box">**片段 {i}** (相似度: {doc.get("score", 0):.2f})<br>{doc["content"][:200]}...</div>',
                        unsafe_allow_html=True)

# 输入框
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                rag = st.session_state.rag_system
                # 检索并生成答案（注意：这里复用已有query方法，但需要返回答案和参考文档）
                # 你可以修改RAGSystem.query方法，使其返回 answer 和 context
                answer, source_docs = rag.query(prompt, k=k, threshold=threshold)
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source_docs": source_docs
                })
            except Exception as e:
                st.error(f"出错了：{str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"抱歉，系统遇到错误：{str(e)}"
                })