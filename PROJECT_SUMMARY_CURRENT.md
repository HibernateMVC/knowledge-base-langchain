# 知识库系统项目总结

## 项目概述
基于通义千问和ElasticSearch的知识库系统，提供智能文档管理和问答功能，集成智能重排序和动态提示生成能力。

## 核心功能
- **单文件上传**: `POST /upload/` - 上传单个文档到知识库
- **批量文件上传**: `POST /upload_batch/` - 支持同时上传多个文档
- **文档搜索**: `POST /search/` - 基于关键词和向量的混合搜索，支持智能重排序
- **智能问答**: `POST /chat/` - 结合上下文的智能对话，支持动态提示生成
- **知识库管理**: 从指定路径添加文档

## 技术架构
- **后端框架**: FastAPI
- **语言模型**: 通义千问（DashScope）qwen-max
- **向量数据库**: ElasticSearch 9.2.1
- **嵌入模型**: 阿里text-embedding-v4
- **文档解析**: LangChain DocumentLoaders
- **提示模板**: LangChain Prompts + 动态提示生成器
- **前端技术**: HTML/CSS/JavaScript
- **重排序模型**: CrossEncoder (可选)

## 项目结构
```
knowledge-base-system/
├── .env                    # 环境配置
├── requirements.txt        # 依赖管理
├── start_server.bat        # Windows启动脚本
├── test_system.py          # 系统测试
├── data/
│   └── uploads/           # 上传文档存储
├── logs/                  # 日志文件
├── README.md              # 项目说明文档
├── PROJECT_SUMMARY.md     # 项目总结报告
├── PROJECT_SUMMARY_CURRENT.md # 当前项目总结
├── RERANKER_MODELS_DOCUMENTATION.md # 重排序功能文档
└── src/
    ├── backend/
    │   └── main.py        # 主服务入口，支持重排序参数
    ├── config/
    │   └── settings.py    # 系统配置文件
    ├── frontend/
    │   └── index.html     # 前端界面，支持拖拽上传和重排序控制
    ├── models/
    │   ├── knowledge_base.py      # 知识库主类，集成重排序功能
    │   └── es_vector_store.py     # ES向量存储，支持混合检索
    ├── utils/
    │   ├── document_loader.py     # 文档加载器
    │   ├── embedding_client.py    # 嵌入客户端
    │   ├── qwen_client.py         # 通义千问客户端
    │   ├── logger.py              # 日志工具
    │   └── log_utils.py           # 详细日志记录
    ├── prompts.py         # 提示词模板和动态提示生成器
    └── debug_server.py    # 调试服务
```

## 特色功能
1. **批量上传支持** - 前后端均支持多文件上传
2. **混合搜索** - 结合关键词匹配和向量相似度
3. **智能重排序** - 使用CrossEncoder模型提升检索质量
4. **动态提示生成** - 根据问题和上下文自适应生成提示
5. **前端友好** - 拖拽上传、实时问答界面、重排序开关
6. **容错处理** - 错误日志记录和异常处理
7. **可扩展性** - 模块化设计便于功能扩展

## 最新改进
- 集成智能重排序功能，支持BGE重排序模型
- 实现动态提示生成器，根据问题类型自适应调整回答策略
- 前端界面增加重排序开关控制
- 优化知识库主类，集成多层检索和问答逻辑
- 改进文档加载器，支持更多格式和更好的分块策略
- 增强日志记录，提供更详细的操作和性能数据

## 运行方式
```bash
# 启动服务
python -m src.backend.main

# 或使用批处理脚本
start_server.bat

# 服务将在 http://localhost:8080 上运行
```

## API文档
- `GET /` - 返回前端界面
- `GET /health/` - 健康检查
- `POST /upload/` - 单文件上传
- `POST /upload_batch/` - 批量文件上传
- `POST /add_document/` - 从路径添加文档
- `POST /search/` - 文档搜索（支持use_reranker参数）
- `POST /chat/` - 智能问答（支持use_reranker参数）