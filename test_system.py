# 系统测试脚本
import os
import sys
import time
import asyncio
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath("."))

from src.models.knowledge_base import KnowledgeBase
from src.config.settings import Config
from src.utils.logger import logger
from src.utils.document_loader import DocumentLoader

def test_basic_components():
    """测试基本组件是否能正常工作"""
    print("="*60)
    print("开始测试知识库系统基本组件")
    print("="*60)
    
    # 测试配置加载
    print("\n1. 测试配置加载...")
    try:
        Config.validate()
        print("   ✓ 配置验证通过")
    except Exception as e:
        print(f"   ✗ 配置验证失败: {e}")
        return False
    
    # 测试日志记录
    print("\n2. 测试日志记录...")
    try:
        logger.info("测试日志记录功能")
        print("   ✓ 日志记录正常")
    except Exception as e:
        print(f"   ✗ 日志记录异常: {e}")
        return False
    
    # 测试知识库初始化
    print("\n3. 测试知识库初始化...")
    try:
        kb = KnowledgeBase()
        kb.initialize()
        print("   ✓ 知识库初始化正常")
    except Exception as e:
        print(f"   ✗ 知识库初始化异常: {e}")
        return False
    
    # 测试文档加载器
    print("\n4. 测试文档加载器...")
    try:
        # 创建测试文本文件
        test_dir = "data/test"
        os.makedirs(test_dir, exist_ok=True)
        
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("这是一个测试文档。\n知识库系统能够处理多种格式的文档。")
        
        docs = DocumentLoader.load_document(test_file)
        print(f"   ✓ 文档加载正常，加载了 {len(docs)} 个文档块")
    except Exception as e:
        print(f"   ✗ 文档加载异常: {e}")
        return False
    
    print("\n✓ 基本组件测试全部通过!")
    return True

def test_embedding_and_storage():
    """测试嵌入和存储功能"""
    print("\n" + "="*60)
    print("开始测试嵌入和存储功能")
    print("="*60)
    
    try:
        kb = KnowledgeBase()
        kb.initialize()
        
        # 测试嵌入功能
        print("\n1. 测试文本嵌入...")
        test_text = "这是用于测试的文本内容"
        embedding = kb.embedding_client.embed_query(test_text)
        print(f"   ✓ 文本嵌入成功，向量维度: {len(embedding)}")
        
        # 测试存储功能
        print("\n2. 测试向量存储...")
        # 添加测试文档到ES
        kb.add_documents([{
            'page_content': test_text,
            'metadata': {'source': 'test', 'type': 'test'}
        }])
        print("   ✓ 文档存储成功")
        
        # 测试搜索功能
        print("\n3. 测试搜索功能...")
        results = kb.search("测试内容", top_k=1)
        print(f"   ✓ 搜索成功，返回 {len(results)} 个结果")
        
        print("\n✓ 嵌入和存储功能测试通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ 嵌入和存储功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_qa_process():
    """测试完整问答流程"""
    print("\n" + "="*60)
    print("开始测试完整问答流程")
    print("="*60)
    
    try:
        kb = KnowledgeBase()
        kb.initialize()
        
        # 添加测试文档
        print("\n1. 添加测试文档...")
        test_docs = [{
            'page_content': "人工智能(Artificial Intelligence, AI)是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            'metadata': {'source': 'ai_introduction', 'type': 'definition'}
        }, {
            'page_content': "机器学习(Machine Learning)是人工智能的一个子领域，它使计算机能够在不进行明确编程的情况下学习和改进。",
            'metadata': {'source': 'ml_definition', 'type': 'definition'}
        }]
        
        kb.add_documents(test_docs)
        print("   ✓ 测试文档添加成功")
        
        # 测试问答
        print("\n2. 测试问答功能...")
        question = "什么是人工智能？"
        result = kb.ask(question, top_k=2)
        
        print(f"   问题: {question}")
        print(f"   答案: {result['answer'][:100]}...")
        print(f"   搜索到 {len(result['sources'])} 个相关文档")
        
        print("\n✓ 完整问答流程测试通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ 完整问答流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_system_test():
    """运行完整系统测试"""
    print("开始运行知识库系统完整测试...")
    
    tests = [
        ("基本组件测试", test_basic_components),
        ("嵌入和存储测试", test_embedding_and_storage),
        ("完整问答流程测试", test_full_qa_process)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-'*60}")
        print(f"运行测试: {test_name}")
        print('-'*60)
        
        if test_func():
            passed_tests += 1
            print(f"\n✓ {test_name} 通过")
        else:
            print(f"\n✗ {test_name} 失败")
    
    print(f"\n{'='*60}")
    print(f"测试完成! 通过: {passed_tests}/{total_tests}")
    print('='*60)
    
    if passed_tests == total_tests:
        print("🎉 所有测试均通过! 知识库系统可以正常运行。")
        return True
    else:
        print("⚠️  部分测试失败，请检查错误信息并修复问题。")
        return False

if __name__ == "__main__":
    success = run_system_test()
    if success:
        print("\n系统已准备好运行。请按照以下步骤启动:")
        print("1. 确保ElasticSearch服务正在运行")
        print("2. 设置环境变量: DASHSCOPE_API_KEY")
        print("3. 运行: python -m src.backend.main")
        print("4. 打开浏览器访问: http://localhost:8000")
    else:
        print("\n系统测试失败，无法启动服务。")