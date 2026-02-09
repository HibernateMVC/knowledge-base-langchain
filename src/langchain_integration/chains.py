"""LangChain Chains实现"""
from typing import Dict, Any, List, Optional
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import BaseOutputParser
from .qwen_model import QwenLLMWrapper
from .es_vector_store_wrapper import ElasticSearchVectorStore
from ..prompts import LangChainPrompts, StepByStepAnalysis
import json
import re
from src.utils.logger import logger


class AnswerTypeClassifier:
    """答案类型分类器"""
    
    @staticmethod
    def determine_answer_type(question: str) -> str:
        """
        根据问题内容判断答案类型
        
        Args:
            question: 问题文本
            
        Returns:
            答案类型 ("name", "number", "boolean", "names", "string")
        """
        question_lower = question.lower()
        
        # 数字类型关键词
        number_keywords = ["多少", "金额", "数值", "数量", "比例", "百分比", "率", "收入", "利润", "资产", "负债", 
                          "销售额", "成本", "费用", "投资", "市值", "股价", "收益", "产值", "产量", "销量"]
        
        # 名称类型关键词
        name_keywords = ["谁", "哪个", "哪位", "什么人", "姓名", "名字", "叫什么", "称谓", "职务", "职位", "角色"]
        
        # Boolean类型关键词
        boolean_keywords = ["是否", "有没有", "是否存在", "能否", "可否", "是不是", "有没有", "是否具备", "是否拥有"]
        
        # Names类型关键词
        names_keywords = ["哪些", "哪些人", "几个人", "都有谁", "都包括", "分别", "列表", "清单", "所有", "多个"]
        
        # 检查数字类型
        if any(keyword in question_lower for keyword in number_keywords):
            return "number"
        
        # 检查名称类型
        if any(keyword in question_lower for keyword in name_keywords):
            return "name"
        
        # 检查Boolean类型
        if any(keyword in question_lower for keyword in boolean_keywords):
            return "boolean"
        
        # 检查Names类型
        if any(keyword in question_lower for keyword in names_keywords):
            return "names"
        
        # 默认为字符串类型
        return "string"


class StructuredOutputParser(BaseOutputParser[Dict[str, Any]]):
    """结构化输出解析器"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        解析模型输出的JSON格式
        
        Args:
            text: 模型输出文本
            
        Returns:
            解析后的字典
        """
        try:
            # 首先尝试处理可能包含markdown代码块的响应
            processed_text = text.strip()
            
            # 如果文本包含markdown代码块，提取其中的内容
            if "```" in processed_text:
                # 找到第一个```之后的内容
                lines = processed_text.split('\n')
                inside_json = False
                json_lines = []
                
                for line in lines:
                    if line.strip().startswith('```json') or line.strip() == '```':
                        if not inside_json and '```json' in line:
                            inside_json = True
                            continue
                        elif inside_json and line.strip() == '```':
                            inside_json = False
                            continue
                    if inside_json:
                        json_lines.append(line)
                
                if json_lines:
                    processed_text = '\n'.join(json_lines).strip()
            
            # 第一步：尝试直接解析整个文本
            try:
                return json.loads(processed_text)
            except json.JSONDecodeError:
                pass  # 继续尝试其他方法
            
            # 第二步：尝试提取花括号内的内容
            start_idx = processed_text.find("{")
            end_idx = processed_text.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_candidate = processed_text[start_idx:end_idx+1]
                
                # 尝试解析提取的内容
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试清理常见格式错误
                    cleaned_json = json_candidate
                    
                    # 移除尾部多余的逗号
                    cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
                    
                    # 替换可能造成问题的特殊字符
                    cleaned_json = cleaned_json.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                    
                    try:
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError:
                        pass  # 继续尝试其他方法
            
            # 第三步：如果上述尝试都失败，尝试使用正则表达式提取键值对
            # 查找常见的JSON键
            result = {}
            
            # 提取 step_by_step_analysis
            step_match = re.search(r'"step_by_step_analysis"\s*:\s*"([^"]*)"', processed_text)
            if step_match:
                result['step_by_step_analysis'] = step_match.group(1).replace('\\n', '\n')
            else:
                # 尝试查找非转义的引号
                step_alt_match = re.search(r'"step_by_step_analysis"[^}]*?"([^"]*)"', processed_text)
                if step_alt_match:
                    result['step_by_step_analysis'] = step_alt_match.group(1).replace('\\n', '\n')
            
            # 提取 reasoning_summary
            summary_match = re.search(r'"reasoning_summary"\s*:\s*"([^"]*)"', processed_text)
            if summary_match:
                result['reasoning_summary'] = summary_match.group(1)
            else:
                summary_alt_match = re.search(r'"reasoning_summary"[^}]*?"([^"]*)"', processed_text)
                if summary_alt_match:
                    result['reasoning_summary'] = summary_alt_match.group(1)
            
            # 提取 relevant_pages
            pages_match = re.search(r'"relevant_pages"\s*:\s*\[([^\]]*)\]', processed_text)
            if pages_match:
                try:
                    pages_str = "[" + pages_match.group(1) + "]"
                    result['relevant_pages'] = json.loads(pages_str)
                except:
                    # 如果列表解析失败，尝试手动解析
                    page_numbers = re.findall(r'\d+', pages_match.group(1))
                    result['relevant_pages'] = [int(p) for p in page_numbers]
            else:
                # 查找列表模式的替代方案
                pages_alt_match = re.search(r'\[([0-9,\s]+)\]', processed_text)
                if pages_alt_match:
                    page_numbers = re.findall(r'\d+', pages_alt_match.group(1))
                    result['relevant_pages'] = [int(p) for p in page_numbers if p]
            
            # 提取 final_answer
            final_match = re.search(r'"final_answer"\s*:\s*"([^"]*)"', processed_text)
            if final_match:
                result['final_answer'] = final_match.group(1)
            else:
                final_alt_match = re.search(r'"final_answer"[^}]*?"([^"]*)"', processed_text)
                if final_alt_match:
                    result['final_answer'] = final_alt_match.group(1)
            
            # 如果至少提取到了一个字段，返回结果
            if result:
                # 补全缺失的字段
                if 'step_by_step_analysis' not in result:
                    result['step_by_step_analysis'] = "未找到分步分析"
                if 'reasoning_summary' not in result:
                    result['reasoning_summary'] = "未找到推理摘要"
                if 'relevant_pages' not in result:
                    result['relevant_pages'] = []
                if 'final_answer' not in result:
                    result['final_answer'] = processed_text
                return result
            else:
                # 如果所有方法都失败，返回默认结构
                return {
                    "step_by_step_analysis": f"无法解析响应格式，原始内容: {processed_text[:200]}...",
                    "reasoning_summary": "响应格式异常",
                    "relevant_pages": [],
                    "final_answer": processed_text
                }
                
        except Exception as e:
            logger.error(f"解析过程中遇到未知错误: {str(e)}")
            logger.error(f"响应内容: {text}")
            return {
                "step_by_step_analysis": f"解析出错: {str(e)}",
                "reasoning_summary": "解析过程中遇到错误",
                "relevant_pages": [],
                "final_answer": text
            }


class RAGChain:
    """RAG链实现"""
    
    def __init__(self, llm: QwenLLMWrapper, vector_store: ElasticSearchVectorStore):
        self.llm = llm
        self.vector_store = vector_store
        self.answer_type_classifier = AnswerTypeClassifier()
        self.output_parser = StructuredOutputParser()
    
    def __call__(
        self, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行RAG链
        
        Args:
            inputs: 输入参数
            
        Returns:
            输出结果
        """
        question = inputs["question"]
        top_k = inputs.get("top_k", 5)
        use_reranker = inputs.get("use_reranker", True)
        reranker_model = inputs.get("reranker_model", "default")

        # 步骤1: 搜索相关文档
        search_results = self.vector_store.hybrid_search(
            query_text=question,
            top_k=top_k,
            use_reranker=use_reranker,
            reranker_model=reranker_model
        )
        
        # 构建上下文
        context_parts = []
        sources = []

        for result in search_results:
            context_parts.append(result['content'])
            # 优先使用重排序得分，如果没有则使用混合得分
            score = result.get('rerank_score', result.get('hybrid_score', 0.0))
            sources.append({
                'id': result['id'],
                'content': result['content'][:200] + "...",
                'score': score,
                'original_score': result.get('hybrid_score', 0.0),  # 保留原始得分
                'rerank_position': result.get('rerank_position'),   # 重排序位置
                'original_position': result.get('original_position'), # 原始位置
                'use_reranker': use_reranker  # 记录是否使用了重排序
            })
        
        context = "\n\n".join(context_parts)
        
        # 步骤2: 根据问题类型确定使用哪种提示
        answer_type = self.answer_type_classifier.determine_answer_type(question)
        
        # 选择相应的提示模板
        if answer_type == "name":
            prompt_template = LangChainPrompts.get_adaptive_name_qa_prompt()
        elif answer_type == "number":
            prompt_template = LangChainPrompts.get_adaptive_number_qa_prompt()
        elif answer_type == "boolean":
            prompt_template = LangChainPrompts.get_adaptive_boolean_qa_prompt()
        elif answer_type == "names":
            prompt_template = LangChainPrompts.get_adaptive_names_qa_prompt()
        else:  # 默认为 string
            prompt_template = LangChainPrompts.get_adaptive_string_qa_prompt()
        
        # 格式化提示
        formatted_prompt = prompt_template.format(context=context, question=question)
        
        try:
            # 步骤3: 调用语言模型
            response = self.llm.invoke(formatted_prompt)
            
            # 步骤4: 解析结构化响应
            structured_response = self.output_parser.parse(response)
            
            # 步骤5: 提取最终答案
            final_answer = structured_response.get('final_answer', '未能从上下文中找到相关信息。')
            
            return {
                'answer': final_answer,
                'sources': sources,
                'search_results': search_results,
                'structured_response': structured_response,
                'answer_type': answer_type
            }
            
        except Exception as e:
            logger.error(f"结构化问答失败: {str(e)}，回退到基础问答")
            
            # 如果结构化问答失败，回退到基础问答
            basic_prompt = LangChainPrompts.basic_rag_prompt.format(context=context, question=question)
            answer = self.llm.invoke(basic_prompt)
            
            return {
                'answer': answer,
                'sources': sources,
                'search_results': search_results,
                'structured_response': {},
                'answer_type': 'fallback'
            }


class QuestionRephraseChain:
    """问题重写链实现"""
    
    def __init__(self, llm: QwenLLMWrapper):
        self.llm = llm
        self.output_parser = StructuredOutputParser()
    
    def __call__(
        self, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行问题重写链
        
        Args:
            inputs: 输入参数
            
        Returns:
            输出结果
        """
        question = inputs["question"]
        companies = inputs["companies"]
        
        # 检查LangChainPrompts中是否存在rephrase_question_prompt
        # 如果不存在，我们使用动态提示生成器或者手动构建
        try:
            from ..prompts import RephrasedQuestionsPrompt
            # 使用原始提示结构构建
            companies_str = ", ".join([f'"{comp}"' for comp in companies])
            user_content = f"原始比较问题：'{question}'\n\n涉及公司：{companies_str}"
            formatted_prompt = f"{RephrasedQuestionsPrompt.system_prompt}\n\n{user_content}"
        except:
            # 回退到简单的提示
            companies_str = ", ".join([f'"{comp}"' for comp in companies])
            formatted_prompt = f"你是一个问题重写系统。将比较问题'{question}'拆解为针对每个公司的独立问题。涉及公司：{companies_str}。请返回JSON格式：{{'questions': [{{'company_name': '...', 'question': '...'}}]}}"
        
        try:
            # 调用语言模型
            response = self.llm.invoke(formatted_prompt)
            
            # 解析响应
            parsed_response = self.output_parser.parse(response)
            rephrased_questions = parsed_response.get('questions', [])
            
            return {
                'rephrased_questions': rephrased_questions
            }
            
        except Exception as e:
            logger.error(f"问题重写失败: {str(e)}")
            raise


class ComparativeAnswerChain:
    """比较类问题最终答案链实现"""
    
    def __init__(self, llm: QwenLLMWrapper):
        self.llm = llm
        self.output_parser = StructuredOutputParser()
    
    def __call__(
        self, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行比较类问题最终答案链
        
        Args:
            inputs: 输入参数
            
        Returns:
            输出结果
        """
        context = inputs["context"]
        question = inputs["question"]
        
        # 检查LangChainPrompts中是否存在comparative_answer_prompt
        # 如果不存在，我们使用动态提示生成器或者手动构建
        try:
            from ..prompts import ComparativeAnswerPrompt
            # 使用原始提示结构构建
            formatted_prompt = f"{ComparativeAnswerPrompt.system_prompt}\n\n以下是单个公司的回答：\n\"{context}\"\n\n---\n\n以下是原始比较问题：\n\"{question}\""
        except:
            # 回退到简单的提示
            formatted_prompt = f"你是一个问答系统，基于各公司独立答案给出原始比较问题的最终结论。只能基于已给出的答案，不可引入外部知识。请分步详细推理。\n\n以下是单个公司的回答：\n\"{context}\"\n\n---\n\n以下是原始比较问题：\n\"{question}\"。请返回JSON格式：{{'step_by_step_analysis': '...', 'reasoning_summary': '...', 'relevant_pages': [], 'final_answer': '...'}}"
        
        try:
            # 调用语言模型
            response = self.llm.invoke(formatted_prompt)
            
            # 解析响应
            parsed_response = self.output_parser.parse(response)
            final_answer = parsed_response.get('final_answer', 'N/A')
            
            return {
                'final_answer': final_answer
            }
            
        except Exception as e:
            logger.error(f"比较类问题处理失败: {str(e)}")
            return {
                'final_answer': 'N/A'
            }