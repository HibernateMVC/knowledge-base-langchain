# 后端FastAPI服务主程序 - LangChain增强版
import sys
import os
import time
import traceback

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.langchain_enhanced_kb import LangChainEnhancedKnowledgeBase
from src.utils.logger import logger
from src.utils.log_utils import log_search_results, log_model_interaction
from src.config.settings import Config


# 移除Langfuse追踪功能
LANGFUSE_CLIENT = None
LANGFUSE_ENABLED = False


# 初始化知识库
kb = LangChainEnhancedKnowledgeBase()
kb.initialize()

# 创建FastAPI应用
app = FastAPI(
    title="知识库系统", 
    description="基于通义千问和ElasticSearch的知识库系统", 
    version="1.0.0",
    responses={404: {"description": "未找到"}}  # 添加默认响应定义
)

# 挂载静态文件目录以提供前端页面
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")), name="static")

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
    use_reranker: bool = True
    reranker_model: str = "default"  # 重排序模型类型

class DocumentAddRequest(BaseModel):
    source: str  # 文档路径或URL

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True
    reranker_model: str = "default"  # 重排序模型类型

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
async def root():
    """根路径 - 返回前端页面"""
    logger.info("根路径访问 - 返回前端页面")
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "index.html")
    try:
        with open(frontend_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.info(f"成功读取前端文件: {frontend_path}")
        return HTMLResponse(content=content, status_code=200)
    except FileNotFoundError:
        logger.warning(f"前端文件未找到: {frontend_path}")
        return {"message": "知识库系统API服务运行正常", "frontend_path": frontend_path}
    except UnicodeDecodeError:
        logger.warning(f"前端文件编码错误: {frontend_path}, 尝试使用gbk编码")
        with open(frontend_path, 'r', encoding='gbk') as file:
            content = file.read()
        return HTMLResponse(content=content, status_code=200)

@app.get("/health/")
async def health_check():
    """健康检查"""
    logger.info("健康检查请求")
    return {
        "status": "healthy",
        "message": "知识库系统运行正常"
    }

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """上传单个文档到知识库（保持原有兼容性）"""
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


@app.post("/upload_batch/")
async def upload_batch(files: List[UploadFile] = File(...)):
    """批量上传文档到知识库"""
    start_time = time.time()
    results = []
    
    # 创建上传目录
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    
    for file in files:
        try:
            # 保存上传的文件
            file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # 添加文档到知识库
            logger.info(f"开始处理上传文件: {file.filename}")
            kb.load_and_add_documents(file_path)
            
            # 记录成功结果
            results.append({
                "filename": file.filename,
                "status": "success",
                "message": f"文档 {file.filename} 已成功添加到知识库"
            })
            
            logger.info(f"文件 {file.filename} 上传并添加到知识库成功")
            
        except Exception as e:
            logger.error(f"上传文件 {file.filename} 失败: {str(e)}")
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            
            # 记录失败结果
            results.append({
                "filename": file.filename,
                "status": "failed",
                "message": f"文档 {file.filename} 上传失败: {str(e)}"
            })
    
    response_time = time.time() - start_time
    logger.info(f"批量上传完成，处理时间: {response_time:.2f}秒，成功 {len([r for r in results if r['status'] == 'success'])} 个，失败 {len([r for r in results if r['status'] == 'failed'])} 个")
    
    return {
        "status": "partial_success" if any(r['status'] == 'failed' for r in results) else "success",
        "results": results,
        "total_uploaded": len(results),
        "success_count": len([r for r in results if r['status'] == 'success']),
        "failure_count": len([r for r in results if r['status'] == 'failed']),
        "processing_time": response_time
    }

@app.post("/add_document/")
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
async def search_documents(request: SearchRequest):
    """搜索文档"""
    start_time = time.time()
    try:
        logger.info(f"开始搜索: {request.query[:50]}...")
        results = kb.search(request.query, request.top_k, use_reranker=request.use_reranker, reranker_model=request.reranker_model)
        
        response_time = time.time() - start_time
        logger.info(f"搜索完成，返回 {len(results)} 个结果，处理时间: {response_time:.2f}秒")
        log_search_results(request.query, results)
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "use_reranker": request.use_reranker,
            "reranker_model": request.reranker_model,
            "response_time": response_time
        }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"搜索失败，处理时间: {response_time:.2f}秒")
        logger.error(f"搜索失败: {str(e)}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.post("/chat/")
async def chat_with_kb(request: QueryRequest):
    """与知识库对话"""
    start_time = time.time()
    try:
        logger.info(f"开始处理问答请求: {request.question[:50]}...")
        
        result = kb.ask(request.question, request.top_k, use_reranker=request.use_reranker, reranker_model=request.reranker_model)
        
        response_time = time.time() - start_time
        logger.info(f"问答处理完成，响应时间: {response_time:.2f}秒")
        log_model_interaction(request.question, result['answer'])
        
        return {
            "question": request.question,
            "answer": result['answer'],
            "sources": result['sources'],
            "search_results": result['search_results'],
            "structured_response": result.get('structured_response', {}),
            "response_time": response_time,
            "use_reranker": request.use_reranker,
            "reranker_model": request.reranker_model
        }
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"问答处理失败，响应时间: {response_time:.2f}秒")
        logger.error(f"问答失败: {str(e)}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=Config.HOST, 
        port=8080,  # 更改为8080端口避免冲突
        log_level=Config.LOG_LEVEL.lower()
    )