# 重排序功能文档

## 概述
本系统支持重排序功能，用于提高检索结果的相关性。系统使用高兼容性重排序器，专注于使用已安装的模型，避免网络连接问题。

## 支持的模型

### 默认模型
- **CrossEncoder模型**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - 轻量级模型，速度较快
  - 需要安装sentence-transformers库
  - 适合各种网络环境

## 安装依赖

要启用重排序功能，需要安装以下依赖：

```bash
# 安装所有依赖（包括重排序功能）
pip install -r requirements.txt

# 或者单独安装重排序依赖
pip install sentence-transformers

# 如果遇到torch问题，可尝试：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 模型选择机制

系统采用以下策略：
1. 检查sentence-transformers库是否已安装
2. 尝试加载CrossEncoder模型
3. 如果加载失败，会跳过重排序功能，但系统仍然正常运行

## 接口参数

### 搜索接口
```
POST /search/
{
  "query": "搜索查询",
  "top_k": 5,
  "use_reranker": true   // 控制是否使用重排序功能
}
```

### 聊天接口
```
POST /chat/
{
  "question": "问题内容",
  "top_k": 5,
  "use_reranker": true   // 控制是否使用重排序功能
}
```

## 故障排除

如果看到 "sentence-transformers库未安装" 错误：

1. 确认已安装依赖：`pip install sentence-transformers`
2. 检查Python环境是否正确
3. 如果仍有问题，可能需要安装Visual C++ Redistributable

即使重排序功能不可用，系统的核心搜索和问答功能也能正常工作。