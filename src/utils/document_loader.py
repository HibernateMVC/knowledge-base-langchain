# 文档加载器
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from src.utils.logger import logger
from src.utils.chinese_text_splitter import ChineseTextSplitter, create_advanced_chinese_splitter


class DocumentLoader:
    """文档加载器，支持多种格式，并使用优化的中文文本分割策略"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    
    @staticmethod
    def load_document(
        file_path: str, 
        use_advanced_splitting: bool = True,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        根据文件扩展名加载文档，并可选择性地使用高级分割策略
        
        Args:
            file_path: 文件路径
            use_advanced_splitting: 是否使用高级分割策略
            chunk_size: 分割块大小
            chunk_overlap: 分割块重叠大小
        
        Returns:
            文档列表
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        logger.info(f"正在加载文档: {file_path}")
        documents = loader.load()
        logger.info(f"成功加载 {len(documents)} 个原始文档块")
        
        # 添加文件元数据
        for doc in documents:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['source'] = file_path
            doc.metadata['filename'] = os.path.basename(file_path)
        
        # 如果启用了高级分割，则使用优化的中文分割器
        if use_advanced_splitting and len(documents) > 0:
            logger.info(f"使用高级分割策略，块大小: {chunk_size}，重叠: {chunk_overlap}")
            splitter = create_advanced_chinese_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            documents = splitter.split_documents(documents)
            logger.info(f"高级分割完成后，共有 {len(documents)} 个文档块")
        
        return documents
    
    @staticmethod
    def load_documents_from_directory(
        directory: str, 
        use_advanced_splitting: bool = True,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        从目录加载所有支持的文档，并可选择性地使用高级分割策略
        
        Args:
            directory: 目录路径
            use_advanced_splitting: 是否使用高级分割策略
            chunk_size: 分割块大小
            chunk_overlap: 分割块重叠大小
        
        Returns:
            文档列表
        """
        all_documents = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1].lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                    try:
                        docs = DocumentLoader.load_document(
                            file_path, 
                            use_advanced_splitting=use_advanced_splitting,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        all_documents.extend(docs)
                    except Exception as e:
                        logger.warning(f"无法加载文件 {file_path}: {str(e)}")
                        continue
        
        logger.info(f"从目录 {directory} 总共加载了 {len(all_documents)} 个文档块")
        return all_documents