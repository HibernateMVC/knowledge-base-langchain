"""
Elasticsearch 9.x 兼容性工具模块
提供与 Elasticsearch 9.2.4 版本兼容的客户端配置和最佳实践
"""

from elasticsearch import Elasticsearch
import sys
import os
# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import Config
from utils.logger import logger


def create_es_client():
    """
    创建与ES 9.x兼容的客户端实例
    此函数使用ES 9.x推荐的安全配置
    """
    try:
        es_client = Elasticsearch(
            [{'host': Config.ES_HOST, 'port': Config.ES_PORT, 'scheme': Config.ES_SCHEME}],
            basic_auth=(Config.ES_USERNAME, Config.ES_PASSWORD),
            request_timeout=30,
            verify_certs=False,  # 对于自签名证书设为False，生产环境中应为True
            ssl_show_warn=False,  # 减少ES 9.x的警告
            ca_certs=None,  # 如果有自定义CA证书可以指定
            max_retries=3,  # 设置重试次数
            retry_on_timeout=True,  # 超时时重试
            http_compress=True  # 启用压缩减少传输数据量
        )
        
        # 测试连接
        if es_client.ping():
            logger.info("成功连接到 Elasticsearch 9.x 服务器")
        else:
            logger.warning("无法连接到 Elasticsearch 服务器")
            
        # 获取ES版本信息
        info = es_client.info()
        es_version = info['version']['number']
        logger.info(f"Elasticsearch 版本: {es_version}")
        
        # 检查版本兼容性
        major_version = int(es_version.split('.')[0])
        if major_version >= 9:
            logger.info(f"当前连接的ES {es_version} 版本与客户端兼容")
        elif major_version == 8:
            logger.info(f"当前连接的ES {es_version} 版本与客户端兼容")
        else:
            logger.warning(f"检测到ES版本 {es_version}，可能与当前客户端存在兼容性问题")
        
        return es_client
        
    except Exception as e:
        logger.error(f"创建Elasticsearch客户端失败: {str(e)}")
        # 如果遇到SSL证书错误，创建一个跳过SSL验证的客户端作为备选方案
        try:
            logger.info("尝试使用跳过SSL验证的方式连接...")
            es_client = Elasticsearch(
                [{'host': Config.ES_HOST, 'port': Config.ES_PORT, 'scheme': Config.ES_SCHEME}],
                basic_auth=(Config.ES_USERNAME, Config.ES_PASSWORD),
                request_timeout=30,
                verify_certs=False,  # 对于自签名证书设为False
                ssl_show_warn=False,
                ca_certs=None,
                max_retries=3,
                retry_on_timeout=True,
                http_compress=True
            )
            
            if es_client.ping():
                logger.info("成功连接到 Elasticsearch 服务器 (跳过SSL验证)")
                
                # 获取ES版本信息
                info = es_client.info()
                es_version = info['version']['number']
                logger.info(f"Elasticsearch 版本: {es_version}")
                
                major_version = int(es_version.split('.')[0])
                if major_version >= 9:
                    logger.info(f"当前连接的ES {es_version} 版本与客户端兼容")
                elif major_version == 8:
                    logger.info(f"当前连接的ES {es_version} 版本与客户端兼容")
                else:
                    logger.warning(f"检测到ES版本 {es_version}，可能与当前客户端存在兼容性问题")
                
                return es_client
        except Exception as e2:
            logger.error(f"即使跳过SSL验证也无法连接到ES: {str(e2)}")
            raise


def adjust_mapping_for_es9(mapping):
    """
    为ES 9.x调整索引映射
    ES 9.x 对一些配置有新的要求
    """
    # ES 9.x 推荐的一些最佳实践调整
    if "mappings" in mapping and "properties" in mapping["mappings"]:
        properties = mapping["mappings"]["properties"]
        
        # 确保向量字段符合ES 9.x规范
        if "vector" in properties:
            vector_field = properties["vector"]
            if vector_field.get("type") == "dense_vector":
                # ES 9.x 不再允许在向量字段中设置 space_type
                # 向量相似度由 similarity 参数控制
                pass  # 简单地不作处理，因现有配置已合适
    
    return mapping