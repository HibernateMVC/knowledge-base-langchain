# 后端FastAPI服务主程序 - 调试版本
import sys
import os
import time
import traceback

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from pydantic import BaseModel
from typing import List, Optional
from models.knowledge_base import KnowledgeBase
from utils.logger import logger
from utils.log_utils import log_api_call, log_search_results, log_model_interaction
from config.settings import Config


# 初始化知识库 - 使用异常处理
try:
    kb = KnowledgeBase()
    kb.initialize()
    logger.info("知识库初始化成功")
except Exception as e:
    logger.error(f"知识库初始化失败: {str(e)}")
    logger.error(traceback.format_exc())
    raise


# 创建FastAPI应用
app = FastAPI(title="知识库系统", description="基于通义千问和ElasticSearch的知识库系统", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API模型定义
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class DocumentAddRequest(BaseModel):
    source: str  # 文档路径或URL

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("知识库系统启动中...")
    try:
        Config.validate()
        logger.info("配置验证通过")
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        raise

@app.get("/")
@log_api_call
async def root():
    """根路径"""
    logger.info("根路径访问")
    return {"message": "知识库系统API服务运行正常"}

@app.post("/upload/")
@log_api_call
async def upload_file(file: UploadFile = File(...)):
    """上传文档到知识库"""
    start_time = time.time()
    try:
        # 创建上传目录
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        
        # 保存上传的文件
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 添加文档到知识库
        logger.info(f"开始处理上传文件: {file.filename}")
        kb.load_and_add_documents(file_path)
        
        response_time = time.time() - start_time
        logger.info(f"文件 {file.filename} 上传并添加到知识库成功，处理时间: {response_time:.2f}秒")
        
        return {
            "filename": file.filename,
            "status": "success",
            "message": f"文档 {file.filename} 已成功添加到知识库",
            "processing_time": response_time
        }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"上传文件失败，处理时间: {response_time:.2f}秒")
        logger.error(f"上传文件失败: {str(e)}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"上传文件失败: {str(e)}")

@app.post("/add_document/")
@log_api_call
async def add_document(request: DocumentAddRequest):
    """从指定路径添加文档到知识库"""
    start_time = time.time()
    try:
        logger.info(f"开始从 {request.source} 添加文档")
        kb.load_and_add_documents(request.source)
        
        response_time = time.time() - start_time
        logger.info(f"从 {request.source} 添加文档成功，处理时间: {response_time:.2f}秒")
        
        return {
            "status": "success",
            "message": f"文档 {request.source} 已成功添加到知识库",
            "processing_time": response_time
        }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"添加文档失败，处理时间: {response_time:.2f}秒")
        logger.error(f"添加文档失败: {str(e)}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"添加文档失败: {str(e)}")

@app.post("/search/")
@log_api_call
async def search_documents(request: SearchRequest):
    """搜索文档"""
    start_time = time.time()
    try:
        logger.info(f"开始搜索: {request.query[:50]}...")
        results = kb.search(request.query, request.top_k)
        
        response_time = time.time() - start_time
        logger.info(f"搜索完成，返回 {len(results)} 个结果，处理时间: {response_time:.2f}秒")
        log_search_results(request.query, results)
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "response_time": response_time
        }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"搜索失败，处理时间: {response_time:.2f}秒")
        logger.error(f"搜索失败: {str(e)}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.post("/chat/")
@log_api_call
async def chat_with_kb(request: QueryRequest):
    """与知识库对话"""
    start_time = time.time()
    try:
        logger.info(f"开始处理问答请求: {request.question[:50]}...")
        
        result = kb.ask(request.question, request.top_k)
        
        response_time = time.time() - start_time
        logger.info(f"问答处理完成，响应时间: {response_time:.2f}秒")
        log_model_interaction(request.question, result['answer'])
        
        return {
            "question": request.question,
            "answer": result['answer'],
            "sources": result['sources'],
            "search_results": result['search_results'],
            "response_time": response_time
        }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"问答处理失败，响应时间: {response_time:.2f}秒")
        logger.error(f"问答失败: {str(e)}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

@app.get("/health/")
@log_api_call
async def health_check():
    """健康检查"""
    logger.info("健康检查请求")
    return {
        "status": "healthy",
        "message": "知识库系统运行正常"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=Config.HOST, 
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower()
    )