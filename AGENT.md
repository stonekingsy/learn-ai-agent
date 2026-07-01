# AI Agent 核心参考手册

> 掌握 AI Agent 的架构、原理、代码实现和最佳实践

## 目录
1. [Agent 核心架构](#1-agent-核心架构)
2. [LLM 推理引擎](#2-llm-推理引擎)
3. [工具系统实现](#3-工具系统实现)
4. [记忆管理](#4-记忆管理)
5. [完整代码示例](#5-完整代码示例)
6. [常见问题与解决](#6-常见问题与解决)

---

## 1. Agent 核心架构

### 1.1 标准 Agent 架构图

```
┌────────────────────────────────────────────────────────┐
│                  用户交互界面                          │
└────────────┬─────────────────────────────┬─────────────┘
             │                             │
             ↓                             ↑
    ┌────────────────────────────────────────────┐
    │          Agent 控制循环                     │
    ├────────────────────────────────────────────┤
    │  Input Processing                          │
    │      ↓                                      │
    │  Context Retrieval (从记忆/知识库)          │
    │      ↓                                      │
    │  Prompt Construction (构造提示词)           │
    │      ↓                                      │
    │  LLM Inference (LLM 推理)                  │
    │      ↓                                      │
    │  Output Parsing (解析输出)                  │
    │      ↓                                      │
    │  Action Execution (执行动作)                │
    │      ↓                                      │
    │  Result Processing (处理结果)               │
    │      ↓                                      │
    │  Output Generation (生成输出)               │
    └────────────────────────────────────────────┘
             │                             ↑
             └─────────────────────────────┘
```

### 1.2 Agent 的四大支柱

```
           ┌─────────────────┐
           │   LLM 大脑      │
           │ (推理 & 决策)   │
           └────────┬────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ↓           ↓           ↓
    ┌────────┐ ┌────────┐ ┌────────┐
    │ 记忆   │ │ 工具   │ │ 反馈   │
    │ Memory │ │ Tools  │ │ Loop   │
    └────────┘ └────────┘ └────────┘
        │           │           │
        └───────────┼───────────┘
                    │
           ┌────────↓────────┐
           │   环境交互      │
           │ (Environment)   │
           └─────────────────┘
```

### 1.3 Agent 的工作周期

```
Initialize Agent
    ↓
[Loop Start]
    ↓
1. Observe: 观察当前状态和新输入
    ↓
2. Think: LLM 分析和推理
    ↓
3. Plan: 制定下一步计划
    ↓
4. Decide: 选择具体动作
    ├─ 需要调用工具？
    │  ├─ 是 → 调用工具获取结果
    │  └─ 否 → 生成最终答案
    ↓
5. Act: 执行决定的动作
    ↓
6. Observe Results: 观察结果反馈
    ↓
7. Update Memory: 更新记忆（如需要）
    ↓
[Task Complete?]
    ├─ 是 → 返回最终结果
    └─ 否 → [Loop Start]
```

---

## 2. LLM 推理引擎

### 2.1 LLM 调用的基本流程

```python
# 基础 LLM 调用
import openai

# 1. 设置 API 密钥
openai.api_key = "your-api-key"

# 2. 构造消息
messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "2+2等于多少？"}
]

# 3. 调用 LLM
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,
    max_tokens=100
)

# 4. 提取结果
answer = response.choices[0].message.content
print(answer)
```

### 2.2 构造高效 Prompt

**Prompt 的四层结构**

```
┌─────────────────────────────────────┐
│  第1层：背景信息（Context）         │
│  - 你是谁、背景、场景               │
├─────────────────────────────────────┤
│  第2层：任务指令（Task）            │
│  - 明确的任务和目标                 │
├─────────────────────────────────────┤
│  第3层：示例（Examples）            │
│  - Few-shot 示例，展示期望格式      │
├─────────────────────────────────────┤
│  第4层：约束（Constraints）         │
│  - 输出格式、限制条件               │
└─────────────────────────────────────┘
```

**Prompt 模板示例**

```python
SYSTEM_PROMPT = """
你是一个专业的数据分析助手。
你的目标是帮助用户分析数据并提供洞察。

# 行为准则
1. 提供客观、基于数据的分析
2. 使用简洁清晰的语言
3. 如果数据不足，说明局限性
4. 提供可行的建议

# 输出格式
使用以下结构组织答案：
- 数据概览
- 关键发现
- 分析洞察
- 建议行动
"""

USER_PROMPT = """
分析这些销售数据（下面提供）：

{data}

请特别关注：
1. 哪些产品表现最好？
2. 销售趋势是什么？
3. 有什么异常情况？
"""
```

### 2.3 链式思考（Chain-of-Thought）

```python
def chain_of_thought_prompt(question):
    """
    使用链式思考提升推理质量
    """
    system = "你是一个逻辑推理助手。"
    
    user = f"""
    请一步步思考以下问题：
    
    问题：{question}
    
    请按以下步骤回答：
    1. 理解问题的关键要素
    2. 识别需要的信息
    3. 逐步推导过程
    4. 验证答案的合理性
    5. 给出最终答案
    
    采用"思考过程 → 结论"的格式。
    """
    
    return system, user
```

---

## 3. 工具系统实现

### 3.1 工具定义

```python
# 工具定义示例
tools = [
    {
        "name": "calculate",
        "description": "执行数学计算操作",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 '2+3*4'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_web",
        "description": "在网络上搜索信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询语句"
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认5"
                }
            },
            "required": ["query"]
        }
    }
]
```

### 3.2 工具执行引擎

```python
import json
from typing import Any, Dict

class ToolExecutor:
    """工具执行管理器"""
    
    def __init__(self):
        self.tools = {}
        self.register_default_tools()
    
    def register_tool(self, name: str, func):
        """注册工具"""
        self.tools[name] = func
    
    def register_default_tools(self):
        """注册默认工具"""
        self.register_tool("calculator", self._calculator)
        self.register_tool("web_search", self._web_search)
        self.register_tool("python_exec", self._python_exec)
    
    def execute(self, tool_name: str, **kwargs) -> str:
        """执行工具"""
        if tool_name not in self.tools:
            return f"错误：工具 {tool_name} 不存在"
        
        try:
            result = self.tools[tool_name](**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行错误：{str(e)}"
    
    def _calculator(self, expression: str) -> float:
        """计算器工具"""
        try:
            return eval(expression)
        except:
            return "计算错误"
    
    def _web_search(self, query: str, num_results: int = 5) -> Dict:
        """网络搜索工具（示例）"""
        # 实际实现可使用 Google API、Bing API 等
        return {
            "query": query,
            "results": [],
            "status": "not_implemented"
        }
    
    def _python_exec(self, code: str) -> str:
        """Python 代码执行（需谨慎）"""
        try:
            # 只在沙箱环境执行
            result = eval(code)
            return str(result)
        except Exception as e:
            return f"执行错误：{str(e)}"
```

### 3.3 工具调用流程

```python
class Agent:
    """基础 Agent 实现"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.executor = ToolExecutor()
    
    def run(self, user_input: str) -> str:
        """主执行循环"""
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 1. 调用 LLM 进行推理
            response = self.llm.complete(
                messages=self.build_messages(user_input),
                tools=self.tools
            )
            
            # 2. 检查是否需要调用工具
            if response.tool_calls:
                tool_results = []
                
                # 3. 执行所有工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    # 执行工具
                    result = self.executor.execute(tool_name, **tool_args)
                    
                    tool_results.append({
                        "tool_name": tool_name,
                        "result": result
                    })
                
                # 4. 将结果反馈给 LLM
                user_input = self.format_tool_results(tool_results)
            else:
                # LLM 直接返回答案，不需要工具
                return response.content
        
        return "达到最大迭代次数"
    
    def build_messages(self, user_input: str) -> list:
        """构建消息列表"""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
    
    def format_tool_results(self, results: list) -> str:
        """格式化工具结果"""
        formatted = "工具执行结果：\n"
        for result in results:
            formatted += f"- {result['tool_name']}: {result['result']}\n"
        formatted += "\n请基于这些结果继续处理用户的请求。"
        return formatted
```

---

## 4. 记忆管理

### 4.1 对话历史管理

```python
class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(self, max_messages: int = 10):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        
        # 保持消息数量在限制内
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_context(self) -> str:
        """获取对话上下文"""
        context = "对话历史：\n"
        for msg in self.messages[-5:]:  # 最后5条消息
            context += f"[{msg['role']}]: {msg['content']}\n"
        return context
    
    def clear(self):
        """清空历史"""
        self.messages = []
```

### 4.2 长期记忆与向量存储

```python
import numpy as np
from typing import List

class LongTermMemory:
    """长期记忆系统"""
    
    def __init__(self, embedding_model):
        self.embeddings = []
        self.memories = []
        self.embedding_model = embedding_model
    
    def remember(self, content: str, importance: float = 1.0):
        """保存记忆"""
        # 向量化内容
        embedding = self.embedding_model.embed(content)
        
        self.memories.append({
            "content": content,
            "embedding": embedding,
            "importance": importance,
            "timestamp": datetime.now()
        })
    
    def recall(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关记忆"""
        if not self.memories:
            return []
        
        # 向量化查询
        query_embedding = self.embedding_model.embed(query)
        
        # 计算相似度
        similarities = []
        for memory in self.memories:
            similarity = self._cosine_similarity(
                query_embedding, 
                memory["embedding"]
            )
            similarities.append(similarity)
        
        # 获取 top-k 记忆
        top_indices = np.argsort(similarities)[-top_k:]
        return [self.memories[i]["content"] for i in top_indices]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 5. 完整代码示例

### 5.1 最小化 Agent 实现

```python
import openai

class MinimalAgent:
    """最小化的 Agent 实现"""
    
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.conversation = []
    
    def chat(self, user_message: str) -> str:
        """单轮对话"""
        self.conversation.append({
            "role": "user",
            "content": user_message
        })
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.conversation
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def multi_turn_conversation(self):
        """多轮对话示例"""
        while True:
            user_input = input("你：")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            response = self.chat(user_input)
            print(f"助手：{response}\n")

# 使用示例
if __name__ == "__main__":
    agent = MinimalAgent("your-api-key")
    agent.multi_turn_conversation()
```

### 5.2 完整的工具使用 Agent

```python
import openai
import json

class ToolUsingAgent:
    """具有工具调用能力的 Agent"""
    
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.tools = [
            {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
        self.conversation = []
    
    def get_weather(self, location: str) -> dict:
        """模拟获取天气（实际应调用真实API）"""
        return {
            "location": location,
            "temperature": 20,
            "condition": "晴天",
            "humidity": 60
        }
    
    def process_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """处理工具调用"""
        if tool_name == "get_current_weather":
            result = self.get_weather(tool_input["location"])
            return json.dumps(result, ensure_ascii=False)
        return "工具不存在"
    
    def run(self, user_query: str) -> str:
        """执行 Agent"""
        self.conversation.append({
            "role": "user",
            "content": user_query
        })
        
        # 第一次调用 LLM
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.conversation,
            tools=self.tools,
            tool_choice="auto"
        )
        
        # 处理响应
        if response.choices[0].finish_reason == "tool_calls":
            # 有工具调用
            tool_calls = response.choices[0].message.tool_calls
            
            # 添加助手消息
            self.conversation.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
            
            # 处理每个工具调用
            for tool_call in tool_calls:
                tool_result = self.process_tool_call(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            # 第二次调用 LLM 获取最终答案
            final_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=self.conversation
            )
            
            final_answer = final_response.choices[0].message.content
            self.conversation.append({
                "role": "assistant",
                "content": final_answer
            })
            
            return final_answer
        else:
            # 直接返回答案
            answer = response.choices[0].message.content
            self.conversation.append({
                "role": "assistant",
                "content": answer
            })
            return answer
```

---

## 6. 常见问题与解决

### 6.1 问题：Agent 陷入无限循环

**症状**
- Agent 不断调用相同的工具
- 不能完成任务

**解决方案**
```python
# 添加最大迭代次数限制
MAX_ITERATIONS = 10

for iteration in range(MAX_ITERATIONS):
    # ... Agent 逻辑
    if iteration >= MAX_ITERATIONS - 1:
        return "达到最大迭代次数，任务未完成"
```

### 6.2 问题：Token 消耗过多

**症状**
- API 成本很高
- 响应慢

**解决方案**
```python
# 1. 压缩对话历史
def compress_history(messages, max_messages=5):
    return messages[-max_messages:]

# 2. 使用流式响应
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    stream=True  # 流式响应
)

# 3. 动态选择模型
if simple_task:
    model = "gpt-3.5-turbo"  # 便宜
else:
    model = "gpt-4"  # 更好
```

### 6.3 问题：工具调用失败

**症状**
- 工具参数错误
- 工具执行异常

**解决方案**
```python
# 1. 参数验证
def validate_tool_params(tool_name, params):
    # 检查必要参数
    if tool_name == "search" and "query" not in params:
        return False, "缺少 query 参数"
    return True, ""

# 2. 错误处理
try:
    result = execute_tool(tool_name, **params)
except Exception as e:
    result = f"工具执行错误：{str(e)}"

# 3. 反馈给 LLM
agent.feedback(f"工具 {tool_name} 执行失败：{result}")
```

### 6.4 问题：输出质量差

**症状**
- 答案不准确
- 答案不相关

**解决方案**
```python
# 1. 改进 Prompt
better_prompt = """
请详细回答以下问题。

问题：{question}

要求：
1. 解释你的推理过程
2. 提供具体的例子
3. 指出任何假设或局限
"""

# 2. 使用 Few-shot learning
examples = [
    {"input": "...", "output": "..."},
    {"input": "...", "output": "..."}
]

# 3. 增加温度参数以获得多样性
response = openai.ChatCompletion.create(
    temperature=0.7,  # 增加创意
    top_p=0.9
)

# 4. 自我纠正
initial_response = get_response()
self_critique = llm.critique(initial_response)
if needs_improvement:
    final_response = llm.improve(initial_response, self_critique)
```

---

## 7. 性能优化指南

### 7.1 响应时间优化

| 优化方式 | 效果 | 难度 |
|--------|------|------|
| 使用流式响应 | ⭐⭐⭐⭐ | 容易 |
| 并行工具调用 | ⭐⭐⭐ | 中等 |
| 缓存常见查询 | ⭐⭐⭐⭐ | 中等 |
| 使用快速模型 | ⭐⭐⭐ | 容易 |

### 7.2 成本优化

```python
# 策略1：分层模型
def choose_model(task_complexity):
    if task_complexity < 3:
        return "gpt-3.5-turbo"  # 最便宜
    elif task_complexity < 7:
        return "gpt-4-turbo"    # 平衡
    else:
        return "gpt-4"          # 最强

# 策略2：缓存
cache = {}

def smart_query(query):
    if query in cache:
        return cache[query]
    result = llm.query(query)
    cache[query] = result
    return result

# 策略3：批处理
def batch_process(queries, batch_size=10):
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)
    return results
```

---

## 总结清单

✅ **基础概念**
- [ ] 理解 Agent 的 4 层架构
- [ ] 掌握 Sense-Think-Act-Learn 循环
- [ ] 理解 LLM 的角色

✅ **关键技术**
- [ ] 能构造高效 Prompt
- [ ] 能设计工具系统
- [ ] 能管理对话历史

✅ **实现能力**
- [ ] 能实现基础 Agent
- [ ] 能集成工具调用
- [ ] 能管理记忆

✅ **优化能力**
- [ ] 能识别性能瓶颈
- [ ] 能优化成本
- [ ] 能改进输出质量

---

*AGENT.md 更新时间：2026-06-15*
