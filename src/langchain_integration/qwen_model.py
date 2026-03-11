"""LangChain兼容的通义千问模型包装器"""
from typing import Any, Dict, List, Optional, Union, Iterator, Mapping
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import GenerationChunk, LLMResult
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.runnables import RunnableConfig
import dashscope
from src.config.settings import Config
from src.utils.logger import logger


class QwenLLMWrapper(BaseLanguageModel[str]):
    """LangChain兼容的通义千问模型包装器"""
    
    model_name: str = Field(default=Config.QWEN_MODEL_NAME, description="模型名称")
    api_key: str = Field(default=Config.DASHSCOPE_API_KEY, description="API密钥")
    temperature: float = Field(default=0.7, description="温度参数")
    max_tokens: int = Field(default=1000, description="最大token数")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def model_post_init(self, __context):
        """在模型初始化后设置DashScope API密钥"""
        # 确保获取到实际值而不是FieldInfo
        api_key = self.api_key
        if hasattr(api_key, '__class__') and not isinstance(api_key, str):
            api_key = Config.DASHSCOPE_API_KEY
        dashscope.api_key = api_key

    def _get_field_value(self, field_value, default_value):
        """安全获取字段值，避免FieldInfo对象"""
        if hasattr(field_value, '__class__') and not isinstance(field_value, (str, int, float, bool)):
            return default_value
        return field_value
        
    @property
    def _llm_type(self) -> str:
        """返回模型类型标识"""
        return "qwen-langchain-wrapper"
        
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """同步生成方法"""
        generations = []
        
        for prompt in prompts:
            try:
                model_name = self._get_field_value(self.model_name, Config.QWEN_MODEL_NAME)
                temperature = self._get_field_value(self.temperature, 0.7)
                max_tokens = self._get_field_value(self.max_tokens, 2000)
                
                response = dashscope.Generation.call(
                    model=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response.status_code == 200:
                    text = response.output.text
                    generations.append([{'text': text}])  # 修改为正确的格式
                    logger.info(f"模型响应成功: {text[:50]}...")
                else:
                    error_msg = f"API调用失败: {response.code} - {response.message}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except Exception as e:
                logger.error(f"生成失败: {str(e)}")
                raise
                
        return LLMResult(generations=generations)
    
    def _stream(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """流式生成方法"""
        for prompt in prompts:
            try:
                # DashScope目前不支持真正的流式输出，我们模拟单次输出
                model_name = self._get_field_value(self.model_name, Config.QWEN_MODEL_NAME)
                temperature = self._get_field_value(self.temperature, 0.7)
                max_tokens = self._get_field_value(self.max_tokens, 2000)
                
                response = dashscope.Generation.call(
                    model=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response.status_code == 200:
                    text = response.output.text
                    # 模拟流式输出
                    yield GenerationChunk(text=text, generation_info={"finish_reason": "stop"})
                else:
                    error_msg = f"API调用失败: {response.code} - {response.message}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except Exception as e:
                logger.error(f"流式生成失败: {str(e)}")
                raise
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """直接调用方法"""
        try:
            model_name = self._get_field_value(self.model_name, Config.QWEN_MODEL_NAME)
            temperature = self._get_field_value(self.temperature, 0.7)
            max_tokens = self._get_field_value(self.max_tokens, 2000)
            
            response = dashscope.Generation.call(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.status_code == 200:
                response_text = response.output.text
                logger.info(f"收到模型响应: {response_text[:50]}...")
                return response_text
            else:
                error_msg = f"API调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"调用失败: {str(e)}")
            raise
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回识别参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    # 实现缺失的抽象方法以便兼容新版LangChain
    def invoke(self, input: LanguageModelInput, config: Optional[RunnableConfig] = None, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """实现invoke方法"""
        if isinstance(input, str):
            return self._call(input, stop=stop, **kwargs)
        elif isinstance(input, list):
            # 处理消息列表
            # 简单实现：将消息列表转换为字符串
            if all(isinstance(msg, str) for msg in input):
                combined_input = " ".join(input)
                return self._call(combined_input, stop=stop, **kwargs)
            else:
                # 处理BaseMessage类型的列表
                combined_input = self._format_messages_to_string(input)
                return self._call(combined_input, stop=stop, **kwargs)
        else:
            # 尝试转换为字符串
            combined_input = str(input)
            return self._call(combined_input, stop=stop, **kwargs)

    def _format_messages_to_string(self, messages: List[BaseMessage]) -> str:
        """将消息列表格式化为字符串"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_messages.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                formatted_messages.append(f"System: {msg.content}")
            else:
                formatted_messages.append(str(msg.content))
        return "\n".join(formatted_messages)

    def predict(self, text: str, **kwargs: Any) -> str:
        """实现predict方法"""
        return self._call(prompt=text, **kwargs)

    def predict_messages(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """实现predict_messages方法"""
        text = self._format_messages_to_string(messages)
        response_text = self._call(prompt=text, **kwargs)
        return AIMessage(content=response_text)

    def generate_prompt(
        self, 
        prompts: List[str], 
        stop: Optional[List[str]] = None, 
        callbacks: Optional[List] = None, 
        **kwargs: Any
    ) -> LLMResult:
        """实现generate_prompt方法"""
        return self._generate(prompts, stop=stop, **kwargs)

    async def agenerate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[List] = None,
        **kwargs: Any
    ) -> LLMResult:
        """实现异步agenerate_prompt方法"""
        # 同步实现
        return self._generate(prompts, stop=stop, **kwargs)

    async def apredict(self, text: str, **kwargs: Any) -> str:
        """实现异步apredict方法"""
        # 同步实现
        return self._call(prompt=text, **kwargs)

    async def apredict_messages(
        self, 
        messages: List[BaseMessage], 
        **kwargs: Any
    ) -> AIMessage:
        """实现异步apredict_messages方法"""
        # 同步实现
        text = self._format_messages_to_string(messages)
        response_text = self._call(prompt=text, **kwargs)
        return AIMessage(content=response_text)