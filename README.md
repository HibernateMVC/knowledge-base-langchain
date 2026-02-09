# 知识库系统

基于通义千问大模型和ElasticSearch向量数据库的知识库问答系统，支持向量+关键词混合检索和智能重排序功能，完全集成LangChain框架。

## 系统架构

- **大语言模型**: 通义千问系列模型(qwen-max)
- **Embedding模型**: 阿里text-embedding-v4
- **向量数据库**: ElasticSearch 9.2.1
- **后端框架**: FastAPI
- **前端界面**: HTML + CSS + JavaScript
- **检索方式**: 向量+关键词混合检索+智能重排序
- **核心框架**: LangChain (完全集成)
- **文档解析**: LangChain DocumentLoaders

## 功能特点

1. **多格式文档支持**: 支持PDF、Word、TXT格式文档上传
2. **智能问答**: 结合知识库内容进行智能回答
3. **混合检索**: 结合向量相似度和关键词匹配
4. **智能重排序**: 支持BGE重排序提升检索质量
5. **LangChain集成**: 基于LangChain框架构建，提供标准化接口
6. **动态提示生成**: 根据文档类型自适应生成提示词
7. **详细日志**: 完整的操作日志记录
8. **Web界面**: 友好的用户交互界面，支持拖拽上传
9. **批量处理**: 支持批量文档上传功能
10. **实时搜索**: 支持关键词搜索和语义搜索

## 环境要求

- Python 3.8+
- ElasticSearch 9.x
- 通义千问API Key
- 阿里云Embedding API Key

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 文件并重命名为 `.env`，然后填入相应的API密钥：

```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
ES_PASSWORD=your_elasticsearch_password
```

### 3. 启动ElasticSearch

确保ElasticSearch服务正在运行，并可根据需要调整连接配置。

### 4. 启动服务

```bash
# Windows
start_server.bat

# 或者手动启动
python -m src.backend.main
```

服务将在 `http://localhost:8080` 上运行

### 5. 使用前端界面

打开浏览器访问 `http://localhost:8080` 使用图形界面

## API接口

### 单文件上传

```
POST /upload/
Content-Type: multipart/form-data

file: [文档文件]
```

### 批量文件上传

```
POST /upload_batch/
Content-Type: multipart/form-data

files: [文档文件列表]
```

### 添加文档（从路径）

```
POST /add_document/
Content-Type: application/json

{
  "source": "文档路径"
}
```

### 问答接口

```
POST /chat/
Content-Type: application/json

{
  "question": "你的问题",
  "top_k": 5,
  "use_reranker": true,
  "reranker_model": "default"
}
```

### 搜索接口

```
POST /search/
Content-Type: application/json

{
  "query": "搜索词",
  "top_k": 5,
  "use_reranker": true,
  "reranker_model": "default"
}
```

### 健康检查

```
GET /health/
```

## 核心特性

### 完全LangChain集成

系统基于LangChain框架构建，充分利用其生态系统优势：

- **标准化接口**: 符合LangChain标准的LLM、Embeddings和VectorStore接口
- **模块化设计**: 可轻松替换不同的LLM、Embedding模型和向量数据库
- **可组合性**: 支持复杂的链式调用和自定义组件

### 混合检索机制

系统采用向量检索和关键词检索相结合的方式：

- **向量检索**: 基于语义相似度的检索
- **关键词检索**: 基于关键词匹配的检索  
- **混合评分**: 综合两种检索结果，提供更准确的答案
- **智能重排序**: 使用CrossEncoder模型进行结果重排序

### 动态提示生成

基于文档内容自动选择最适合的提示词策略：

- **文档类型识别**: 自动识别财务、法律、技术、学术等文档类型
- **自适应提示**: 根据文档类型生成针对性的提示词
- **结构化输出**: 支持JSON格式的结构化回答

### 文档处理流程

1. 文档上传 → 解析 → 分段
2. 分段向量化 → 存储到ES
3. 查询向量化 → 混合检索 → 智能重排序
4. 动态提示生成 → 生成回答 → 返回结果

### 日志记录

系统提供详细的日志记录功能：

- **操作日志**: 记录所有API调用
- **性能日志**: 记录响应时间
- **错误日志**: 详细的错误信息和堆栈跟踪
- **业务日志**: 记录搜索、问答等业务流程

## 文件结构

```
knowledge-base-system/
├── data/                   # 数据存储目录
│   └── uploads/           # 上传的文档
├── logs/                  # 日志文件
├── src/
│   ├── backend/          # 后端服务
│   │   └── main.py       # 主服务入口，使用LangChain增强版知识库
│   ├── config/           # 配置文件
│   │   └── settings.py   # 系统配置
│   ├── frontend/         # 前端页面
│   │   └── index.html    # 主界面
│   ├── langchain_integration/ # LangChain集成组件
│   │   ├── chains.py     # 各种LangChain链实现
│   │   ├── es_vector_store_wrapper.py # ES向量存储包装器
│   │   ├── qwen_model.py # 通义千问LLM包装器
│   │   └── __init__.py   # 集成模块初始化
│   ├── models/           # 数据模型
│   │   ├── knowledge_base.py      # 传统知识库类（保留兼容性）
│   │   └── es_vector_store.py     # ES向量存储
│   ├── utils/            # 工具函数
│   │   ├── document_loader.py     # 文档加载器
│   │   ├── embedding_client.py    # 嵌入客户端
│   │   ├── qwen_client.py         # 通义千问客户端
│   │   └── logger.py              # 日志工具
│   ├── prompts.py        # 提示词模板和动态提示生成器
│   ├── langchain_enhanced_kb.py   # LangChain增强版知识库主类
│   └── debug_server.py   # 调试服务
├── requirements.txt       # 依赖包列表
├── start_server.bat      # 启动脚本
├── README.md            # 项目说明文档
├── PROJECT_SUMMARY.md   # 项目总结报告
├── PROJECT_SUMMARY_CURRENT.md # 当前项目总结
├── RERANKER_MODELS_DOCUMENTATION.md # 重排序功能文档
└── DOCUMENTATION_CHUNKING_STRATEGY.md # 分块策略文档
```

## 配置说明

### 系统配置

可在 `src/config/settings.py` 中调整各项配置：

```python
# 通义千问配置
DASHSCOPE_API_KEY     # API密钥（用于通义千问和Embedding模型）
QWEN_MODEL_NAME       # 模型名称，默认 qwen-max

# 阿里embedding模型配置
EMBEDDING_MODEL_NAME  # Embedding模型名称，默认 text-embedding-v4

# Elasticsearch配置
ES_HOST               # ES主机地址，默认 localhost
ES_PORT               # ES端口，默认 9200
ES_SCHEME             # 协议，默认 https，也可选 http
ES_USERNAME           # 用户名，默认 elastic
ES_PASSWORD           # 密码
ES_INDEX_NAME         # 索引名称，默认 knowledge_base_index

# 服务配置
HOST                # 服务监听地址，默认 0.0.0.0
PORT                # 服务监听端口，默认 8080
```

## 重排序功能

系统支持智能重排序功能以提升搜索结果质量：

- **默认模型**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **控制参数**: use_reranker (布尔值)
- **模型选择**: reranker_model (目前为默认值)

要启用完整的重排序功能，需要安装相关依赖：
```bash
pip install sentence-transformers
```

## LangChain框架优势

项目全面集成LangChain框架，带来以下优势：

- **组件化**: 易于替换不同模型和服务提供商
- **标准化**: 遵循LangChain接口标准，便于扩展
- **可组合**: 支持复杂的链式调用和自定义组件
- **生态系统**: 充分利用LangChain丰富的工具和组件
- **向后兼容**: 保留了原有功能的同时增加新能力

## 故障排除

### 常见问题

1. **API Key错误**: 确认DASHSCOPE_API_KEY正确设置
2. **ES连接失败**: 检查ElasticSearch服务是否正常运行
3. **依赖安装失败**: 尝试使用国内镜像源
4. **重排序模型下载失败**: 检查网络连接或离线安装模型
5. **LangChain组件问题**: 确保langchain相关依赖正确安装

### 日志查看

日志文件位于 `logs/` 目录，可按日期查看相应日志文件。

### 性能优化

- 调整ES索引参数以提高检索速度
- 合理设置top_k参数平衡准确性和性能
- 定期清理不必要的文档数据
- 启用重排序功能提高结果精度

## 安全注意事项

- API Key请妥善保管，不要泄露
- 生产环境请限制API访问权限
- 建议启用HTTPS加密传输
- 定期更新依赖包以修复安全漏洞