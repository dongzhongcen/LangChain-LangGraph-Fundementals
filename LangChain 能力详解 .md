# LangChain 能力详解

> 本文为个人学习笔记，基于课程资料整理，仅用于个人学习和复习。

## 📑 目录
- [环境配置](#环境配置)
- [快速上手](#快速上手)
- [LangChain核心概念](#langchain核心概念)
- [聊天模型核心能力](#聊天模型核心能力)
- [消息机制](#消息机制)
- [提示词模板](#提示词模板)
- [少样本提示](#少样本提示)
- [输出解析器](#输出解析器)
- [文档加载器](#文档加载器)
- [文本分割器](#文本分割器)
- [文本向量与向量存储](#文本向量与向量存储)
- [检索器](#检索器)
- [RAG实战案例](#rag实战案例)

---

## 环境配置

### Python环境
- Python 3.13

### 所需依赖包
```python
# 第一部分：核心依赖
langchain-openai==0.3.33
langchain==0.3.27
langchain-deepseek==0.1.4
langchain-ollama==0.3.6
langchain-tavily==0.2.12
langchain-chroma==0.2.5
langchain-community==0.3.22
nltk==3.9.2
langchain-redis==0.2.4
unstructured==0.18.15
markdown==3.9
redisvl==0.10.0

# 第二部分：Pinecone（注意安装顺序问题）
pinecone==7.3.0
langchain-pinecone==0.2.12
```

**⚠️ 注意事项**：
- Pinecone相关包需要在学习到向量存储部分再安装
- 安装langchain-pinecone会影响Redis的MMR搜索功能
- 高版本Pinecone可能存在numpy兼容性问题

安装命令：
```bash
pip install -r requirements.txt
```

---

## 快速上手

### 为什么要用LangChain？

原生LLM开发面临的问题：
- 提示词不规范，结果容易出现幻觉
- 模型切换困难
- 输出非结构化，难以与程序接口对接
- 知识陈旧，无法获取实时信息
- 难以连接外部工具和系统

LangChain的核心目标就是解决这些问题，将NLP流程拆解为标准化组件。

### 最简单的对话示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# 1. 定义大模型
model = ChatOpenAI(model="gpt-4o-mini")

# 2. 定义消息列表
messages = [
    SystemMessage(content="Translate the following from English into Chinese"),
    HumanMessage(content="hi!")
]

# 3. 定义输出解析器
parser = StrOutputParser()

# 4. 定义链
chain = model | parser

# 5. 执行链
result = chain.invoke(messages)
print(result)  # 输出：你好！
```

### 关键概念引出

**Runnable接口**：LangChain组件的基础接口，提供：
- **Invoked**：单个输入转换为输出
- **Batched**：批量处理
- **Streamed**：流式传输
- **Inspected**：检查功能
- **Composed**：组合能力

**LCEL (LangChain Expression Language)**：声明式编程方式，通过`|`操作符合并Runnable对象。

```python
# 以下三种写法等价
chain = model | parser
chain = RunnableSequence(first=model, last=parser)
chain = model.pipe(parser)
```

---

## 聊天模型核心能力

### 1. 定义聊天模型

#### 方式1：ChatOpenAI（明确指定）

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,           # 控制随机性，0最保守
    max_tokens=None,         # 最大生成token数
    timeout=None,            # 超时时间
    max_retries=2,           # 最大重试次数
    # api_key="your-key",    # 可从环境变量读取
    # base_url="...",        # API请求基础URL
)
```

#### 方式2：init_chat_model（工厂函数）

```python
from langchain.chat_models import init_chat_model

# 基本用法
gpt_model = init_chat_model("gpt-4o-mini", model_provider="openai")
deepseek_model = init_chat_model("deepseek-chat", model_provider="deepseek")

# 可配置模型
configurable_model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0,
    configurable_fields=("model", "temperature"),
    config_prefix="first"
)

# 动态配置
result = configurable_model.invoke(
    "what's your name",
    config={
        "configurable": {
            "first_model": "deepseek-chat",
            "first_temperature": 0.5
        }
    }
)
```

#### 方式3：本地部署模型（ChatOllama）

```python
from langchain_ollama import ChatOllama

ollama_model = ChatOllama(
    model="deepseek-r1:70b",
    base_url='http://192.168.100.220:11434',
    num_ctx=2048,    # 上下文窗口大小
    num_gpu=1        # GPU数量
)
```

### 2. 工具调用

工具调用让LLM能够与外部世界交互，扩展能力边界。

#### 创建工具的多种方式

**方式1：@tool装饰器（最常用）**

```python
from langchain_core.tools import tool
from typing_extensions import Annotated

# 需要文档字符串
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# 使用Annotated提供参数描述
@tool
def add(
    a: Annotated[int, ..., "First integer"],
    b: Annotated[int, ..., "Second integer"]
) -> int:
    """Add two integers."""
    return a + b
```

**方式2：依赖Pydantic类**

```python
from pydantic import BaseModel, Field

class AddInput(BaseModel):
    """Add two integers."""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

@tool(args_schema=AddInput)
def add(a: int, b: int) -> int:
    return a + b  # 无需文档字符串
```

**方式3：StructuredTool.from_function**

```python
from langchain_core.tools import StructuredTool

def multiply(a: int, b: int) -> int:
    return a * b

calculator_tool = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="两数相乘",
    args_schema=CalculatorInput,
    response_format="content_and_artifact"  # 返回内容和原始数据
)
```

#### 绑定工具

```python
# 绑定工具到模型
tools = [add, multiply]
model_with_tools = model.bind_tools(tools)

# 或强制调用工具
model_with_tools = model.bind_tools(tools, tool_choice="any")
```

#### 完整工具调用流程

```python
from langchain_core.messages import HumanMessage

# 1. 定义消息
messages = [HumanMessage("9乘6等于多少？5加3等于多少？")]

# 2. 第一次调用：获取工具调用指令
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# 3. 执行工具调用
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

# 4. 第二次调用：获取最终答案
result = model.invoke(messages)
print(result.content)  # 输出：9乘6等于54，5加3等于8。
```

#### LangChain内置工具示例：Tavily搜索

```python
from langchain_tavily import TavilySearch

# 配置环境变量 TAVILY_API_KEY
tool = TavilySearch(max_results=4)
model_with_tools = model.bind_tools([tool])

messages = [HumanMessage("中国西安今天的天气怎么样？")]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    tool_msg = tool.invoke(tool_call)
    messages.append(tool_msg)

result = model_with_tools.invoke(messages)
print(result.content)
```

### 3. 结构化输出

将模型输出转换为结构化格式（JSON、Pydantic对象等）。

```python
from pydantic import BaseModel, Field
from typing import Optional

class Joke(BaseModel):
    """给用户讲一个笑话。"""
    setup: str = Field(description="这个笑话的开头")
    punchline: str = Field(description="这个笑话的妙语")
    rating: Optional[int] = Field(default=None, description="评分1-10")

# 绑定结构化输出
structured_model = model.with_structured_output(Joke)
result = structured_model.invoke("给我讲一个关于唱歌的笑话")
print(result)  # 输出Pydantic对象
```

**嵌套结构示例**：

```python
class Data(BaseModel):
    """获取关于笑话的数据。"""
    jokes: List[Joke]

structured_model = model.with_structured_output(Data)
result = structured_model.invoke("分别讲一个关于唱歌和跳舞的笑话")
```

### 4. 流式传输

```python
# 同步流式
chunks = []
for chunk in model.stream("讲一个50字笑话"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)

# 异步流式
import asyncio

async def async_stream():
    async for chunk in model.astream("讲一个50字笑话"):
        print(chunk.content, end="", flush=True)

asyncio.run(async_stream())

# 带解析器的流式
chain = model | StrOutputParser()
for chunk in chain.stream("写一段关于爱情的歌词"):
    print(chunk, end="|", flush=True)
```

#### 自定义流式解析器

```python
from typing import Iterator

def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    buffer = ""
    for chunk in input:
        buffer += chunk
        while "。" in buffer:
            stop_index = buffer.index("。")
            yield [buffer[:stop_index].strip()]
            buffer = buffer[stop_index + 1:]
    yield [buffer.strip()]

chain = model | StrOutputParser() | split_into_list
```

### 5. 使用LangSmith追踪

```python
# 配置环境变量
# LANGSMITH_TRACING="true"
# LANGSMITH_API_KEY="你的LangSmith API Key"

# 任意代码执行后，在LangSmith平台查看追踪信息
```

---

## 消息机制

### 消息类型

| 类型 | 对应角色 | 描述 |
|------|----------|------|
| SystemMessage | system | 系统指令，设定对话基调 |
| HumanMessage | user | 用户输入 |
| AIMessage | assistant | 模型响应 |
| ToolMessage | tool | 工具调用结果 |

### 多轮对话与内存

```python
from langchain_core.messages import HumanMessage, AIMessage

# 手动维护历史消息
messages = [
    HumanMessage(content="Hi! I'm Bob"),
    AIMessage(content="Hello Bob! How can I assist you today?"),
    HumanMessage(content="What's my name?"),
]
model.invoke(messages).pretty_print()
```

### 消息裁剪

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=65,           # 最大token数
    strategy="last",         # 保留最后的消息
    token_counter=model,     # token计数方法
    include_system=True,     # 保留系统消息
    allow_partial=False,     # 不允许拆分消息
    start_on="human",        # 确保第一条消息是human
)

chain = trimmer | model
```

### 消息过滤

```python
from langchain_core.messages import filter_messages

# 按类型过滤
filter_messages(messages, include_types="human")

# 按类型+ID过滤
filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"])
```

### 消息合并

```python
from langchain_core.messages import merge_message_runs

merged = merge_message_runs(messages)
chain = merge_message_runs() | model
```

---

## 提示词模板

### 字符串模板

```python
from langchain_core.prompts import PromptTemplate

# 方式1：from_template
prompt_template = PromptTemplate.from_template("Translate the following into {language}")

# 方式2：直接初始化
prompt_template = PromptTemplate(
    input_variables=["language"],
    template="Translate the following into {language}",
)

result = prompt_template.invoke({"language": "Chinese"})
```

### 聊天消息模板

```python
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate([
    ("system", "Translate the following into {language}."),
    ("user", "{text}")
])

# 实例化
messages = prompt_template.invoke({
    "language": "Chinese",
    "text": "what is your name?"
}).to_messages()

# 链式调用
chain = prompt_template | model | StrOutputParser()
```

### 消息占位符

```python
from langchain_core.prompts import MessagesPlaceholder

prompt_template = ChatPromptTemplate([
    ("system", "你是一个聊天助手"),
    MessagesPlaceholder("msgs")  # 或 ("placeholder", "{msgs}")
])

messages_to_pass = [
    HumanMessage(content="中国首都是哪里？"),
    AIMessage(content="中国首都是北京。"),
    HumanMessage(content="那法国呢？")
]

result = prompt_template.invoke({"msgs": messages_to_pass})
```

### 使用LangChain Hub

```python
from langsmith import Client

client = Client()
prompt = client.pull_prompt("hardkothari/prompt-maker", include_model=True)
chain = prompt | model

while True:
    task = input("你的任务是什么？")
    lazy_prompt = input("你当前的提示是什么？")
    chain.invoke({'lazy_prompt': lazy_prompt, 'task': task}).pretty_print()
```

---

## 少样本提示

### 基本用法

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# 定义示例
examples = [
    {"input": "2 2", "output": "4"},
    {"input": "2 3", "output": "5"},
]

# 定义示例模板
example_prompt = ChatPromptTemplate([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 创建少样本提示
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 组合最终提示
final_prompt = ChatPromptTemplate([
    ("system", "你是一个神奇的数学奇才。"),
    few_shot_prompt,
    ("human", "{input}"),
])

chain = final_prompt | model
chain.invoke({"input": "What is 2 9?"}).pretty_print()  # 输出：11
```

### 推理引导示例

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 字符串模板
example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

# 示例集（包含推理过程）
examples = [
    {
        "question": "李白和杜甫，谁更长寿？",
        "answer": """是否需要后续问题：是的。
后续问题：李白享年多少岁？
中间答案：李白享年61岁。
后续问题：杜甫享年多少岁？
中间答案：杜甫享年58岁。
所以最终答案是：李白"""
    },
    # ... 更多示例
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

prompt_messages = prompt.invoke({
    "input": "《教父》和《星球大战》的导演来自同一个国家吗？"
}).to_messages()
```

### 示例选择器

#### 按长度选择

```python
from langchain_core.example_selectors import LengthBasedExampleSelector

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入的反义词",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
```

#### 按语义相似性选择

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=1,
)
```

#### 按MMR选择（最大边际相关性）

```python
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2,
)
```

---

## 输出解析器

### 文本解析器

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = model | parser
```

### 结构化对象解析器

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
```

### JSON解析器

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
# 或带Pydantic验证
parser = JsonOutputParser(pydantic_object=Joke)
```

其他解析器：XMLOutputParser、YamlOutputParser、CommaSeparatedListOutputParser、EnumOutputParser等。

---

## 文档加载器

### Document对象

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="狗是很好的伴侣，以忠诚和友好而闻名。",
        metadata={"source": "mammal-pets-doc"},
    ),
    # ...
]
```

### 加载PDF

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./Docs/PDF/文档.pdf")
docs = loader.load()  # 每页一个Document

print(f"总页数：{len(docs)}")
print(f"第一页内容：{docs[0].page_content[:200]}")
print(f"第一页元数据：{docs[0].metadata}")
```

### 加载Markdown

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# single模式：整个文档作为一个Document
loader = UnstructuredMarkdownLoader("./Docs/Markdown/文档.md", mode="single")
data = loader.load()

# elements模式：拆分为多个元素
loader = UnstructuredMarkdownLoader("./Docs/Markdown/文档.md", mode="elements")
data = loader.load()  # 包含Title、ListItem、NarrativeText等类型
```

---

## 文本分割器

### 基于字符长度拆分

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",        # 分隔符
    chunk_size=100,          # 目标块大小
    chunk_overlap=20,        # 块重叠大小
    length_function=len,     # 长度计算函数
    is_separator_regex=False,
)

texts = text_splitter.split_documents(documents)
```

### 基于Token长度拆分

```python
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # GPT-4、GPT-3.5-turbo的编码方式
    chunk_size=200,
    chunk_overlap=50,
)
```

### 硬约束长度拆分

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=100,
    chunk_overlap=0,
)
```

### 代码文档拆分

```python
from langchain_text_splitters import PythonCodeTextSplitter

python_splitter = PythonCodeTextSplitter(chunk_size=50, chunk_overlap=0)
python_docs = python_splitter.create_documents([PYTHON_CODE])
```

---

## 文本向量与向量存储

### 嵌入模型

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 嵌入文档列表
texts = [doc.page_content for doc in documents]
documents_vector = embeddings.embed_documents(texts)
print(f"向量维度：{len(documents_vector[0])}")  # 3072维

# 嵌入单个查询
query_vector = embeddings.embed_query("项目中遇到了哪些挑战？")
```

### 内存向量存储

```python
from langchain_core.vectorstores import InMemoryVectorStore

# 初始化
vector_store = InMemoryVectorStore(embedding=embeddings)

# 添加文档
ids = vector_store.add_documents(documents=documents)

# 获取文档
docs = vector_store.get_by_ids(ids[:3])

# 相似性搜索
search_docs = vector_store.similarity_search(query="数据库表怎么设计的？", k=2)

# 元数据过滤
def filter_function(doc: Document) -> bool:
    return doc.metadata.get("source") == "期望的source"

search_docs = vector_store.similarity_search(
    query="数据库表怎么设计的？",
    k=2,
    filter=filter_function
)

# 删除文档
vector_store.delete(ids=ids[:3])
```

### Redis向量存储

```python
from langchain_redis import RedisConfig, RedisVectorStore
from redisvl.query.filter import Tag, Num

# 配置
config = RedisConfig(
    index_name="qa",
    redis_url="redis://localhost:6379",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"},
    ],
)

# 初始化
vector_store = RedisVectorStore(embeddings, config=config)

# 添加文档时添加元数据
for i, doc in enumerate(documents, start=1):
    doc.metadata["category"] = "QA"
    doc.metadata["num"] = i

ids = vector_store.add_documents(documents=documents)

# 相似性搜索带分数
scored_results = vector_store.similarity_search_with_score(
    query="数据库表怎么设计的？",
    k=4
)

# 元数据过滤
filter_condition = (Tag("category") == "qa") & (Num("num") < 50)
scored_results = vector_store.similarity_search_with_score(
    query="数据库表怎么设计的？",
    k=2,
    filter=filter_condition
)

# MMR搜索
mmr_results = vector_store.max_marginal_relevance_search(
    query="数据库表怎么设计的？",
    k=2,
    fetch_k=10,
    filter=filter_condition
)
```

### Pinecone向量存储

```python
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# 初始化Pinecone
pc = Pinecone()
index_name = "qa"

# 创建索引
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# 获取索引
index = pc.Index(index_name)

# 初始化向量存储
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# 添加文档
ids = vector_store.add_documents(documents=documents)

# 相似性搜索
search_docs = vector_store.similarity_search(
    query="数据库表怎么设计的？",
    k=2,
    filter={"category": "QA"}
)

# 删除
vector_store.delete(delete_all=True)  # 全量删除
vector_store.delete(ids=delete_ids)    # 指定ID删除
```

---

## 检索器

### 从向量存储创建检索器

```python
# 基本检索器
retriever = vector_store.as_retriever()
docs = retriever.invoke("数据库表怎么设计的？")

# 配置参数
retriever = vector_store.as_retriever(
    search_type="similarity",  # 或 "mmr", "similarity_score_threshold"
    search_kwargs={"k": 2}
)

# MMR配置
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 10
    }
)
```

### 自定义检索器

```python
from langchain_core.runnables import chain
from typing import List
from langchain_core.documents import Document

@chain
def custom_retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=2)

docs = custom_retriever.invoke("数据库表怎么设计的？")
```

---

## RAG实战案例

### 完整RAG流程

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. 初始化模型
model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 2. 配置向量存储
config = RedisConfig(
    index_name="qa",
    redis_url="redis://192.168.100.238:6379",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"},
    ],
)
vector_store = RedisVectorStore(embeddings, config=config)
retriever = vector_store.as_retriever()

# 3. 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("human", """你是负责回答问题的助手。使用以下检索到的上下文片段来回答问题。
如果你不知道答案，就说你不知道。最多只用三句话，回答要简明扼要。

Question: {question}
Context: {context}
Answer:""")
])

# 4. 文档格式化函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. 构建RAG链
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# 6. 流式执行
for chunk in rag_chain.stream("数据库表怎么设计的？"):
    print(chunk, end="", flush=True)
```

### 交互式RAG问答

```python
while True:
    question = input("\n请输入您的问题（输入'退出'结束）：").strip()
    if question.lower() in ["退出", "quit"]:
        break
    if not question:
        continue
    
    print("回答：", end="", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()
```

---

## 📌 核心概念总结

| 组件 | 作用 | 常用类/方法 |
|------|------|------------|
| **聊天模型** | 与LLM交互 | ChatOpenAI, init_chat_model |
| **消息** | 通信单位 | SystemMessage, HumanMessage, AIMessage |
| **提示词模板** | 动态生成提示词 | PromptTemplate, ChatPromptTemplate |
| **输出解析器** | 结构化输出 | StrOutputParser, PydanticOutputParser |
| **工具调用** | 扩展LLM能力 | @tool, bind_tools() |
| **文档加载器** | 加载各类文档 | PyPDFLoader, UnstructuredMarkdownLoader |
| **文本分割器** | 切分文档 | CharacterTextSplitter, RecursiveCharacterTextSplitter |
| **嵌入模型** | 文本转向量 | OpenAIEmbeddings |
| **向量存储** | 存储和检索向量 | InMemoryVectorStore, RedisVectorStore |
| **检索器** | 统一检索接口 | as_retriever(), 自定义@chain |
| **LCEL** | 声明式链式编程 | `\|` 操作符, RunnableSequence |

---

> **个人学习心得**：
> - LangChain的核心思想是**组件化**和**可组合性**，每个组件都实现Runnable接口
> - LCEL让构建复杂流程变得简单直观
> - RAG是当前最实用的LLM应用模式，掌握好文档加载→分割→嵌入→存储→检索→生成的完整流程
> - 遇到问题时，查看官方文档和源码是最好的学习方式