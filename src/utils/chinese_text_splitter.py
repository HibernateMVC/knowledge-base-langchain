"""
增强的文档分割器，结合LangChain TextSplitter和滑动窗口策略
用于优化中文文档的分块效果
"""
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.logger import logger


class ChineseTextSplitter:
    """
    中文文本分割器，专门为中文文档优化
    结合了LangChain的TextSplitter和滑动窗口策略
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: Optional[callable] = None
    ):
        """
        初始化中文文本分割器
        
        Args:
            chunk_size: 块大小
            chunk_overlap: 块重叠大小（实现滑动窗口效果）
            separators: 分隔符列表，用于分割文本
            length_function: 长度计算函数
        """
        # 默认的中文分隔符，按优先级排列
        if separators is None:
            separators = [
                "\n\n",  # 段落分隔
                "\n",     # 换行符
                "。",      # 中文句号
                "！",      # 中文感叹号
                "？",      # 中文问号
                "；",      # 中文分号
                "；",      # 中文分号（可能出现的其他形式）
                "……",     # 中文省略号
                "…",      # 中文省略号
                ",",      # 英文逗号
                ".",      # 英文句号
                "!",      # 英文感叹号
                "?",      # 英文问号
                ";",      # 英文分号
                " ",      # 空格
                ""
            ]
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.length_function = length_function or len
        
        # 初始化LangChain的RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=self.length_function,
            is_separator_regex=False
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表
        
        Args:
            documents: 输入的文档列表
            
        Returns:
            分割后的文档列表
        """
        logger.info(f"开始分割 {len(documents)} 个文档，块大小: {self.chunk_size}，重叠: {self.chunk_overlap}")
        
        split_docs = []
        
        for doc in documents:
            # 使用LangChain的text_splitter分割单个文档
            doc_splits = self.text_splitter.split_documents([doc])
            
            # 为每个分割添加元数据
            for split_doc in doc_splits:
                # 保留原始元数据
                if not hasattr(split_doc, 'metadata'):
                    split_doc.metadata = {}
                
                # 添加分割相关的元数据
                split_doc.metadata['chunk_size'] = self.chunk_size
                split_doc.metadata['chunk_overlap'] = self.chunk_overlap
                split_doc.metadata['original_length'] = len(doc.page_content)
                
                split_docs.append(split_doc)
        
        logger.info(f"分割完成，得到 {len(split_docs)} 个文档块")
        return split_docs

    def split_text(self, text: str) -> List[str]:
        """
        直接分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            分割后的文本列表
        """
        return self.text_splitter.split_text(text)


# 全局默认分割器实例
default_chinese_splitter = ChineseTextSplitter()


def create_advanced_chinese_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> ChineseTextSplitter:
    """
    创建高级中文分割器的工厂函数
    
    Args:
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        separators: 分隔符列表
        
    Returns:
        配置好的ChineseTextSplitter实例
    """
    return ChineseTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )