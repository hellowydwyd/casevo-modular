# 最佳实践指南

本文档介绍使用 Casevo 框架的最佳实践。

## 代码规范

### 1. 命名规范

- **类名**: 使用 PascalCase，如 `ElectionAgent`, `ResourceModel`
- **函数名**: 使用 snake_case，如 `run_experiment`, `get_memory`
- **常量**: 使用 UPPER_SNAKE_CASE，如 `DEFAULT_API_KEY`
- **私有方法**: 使用单下划线前缀，如 `_internal_method`

### 2. 文档字符串

所有公共类和方法都应该有文档字符串：

```python
class MyAgent(AgentBase):
    """
    我的智能体类
    
    描述智能体的功能和用途。
    
    Attributes:
        agent_id: 智能体唯一标识
        memory: 记忆系统实例
    """
    
    def step(self):
        """
        执行智能体的一步行为
        
        Returns:
            None
        """
        pass
```

### 3. 类型提示

使用类型提示提高代码可读性：

```python
from typing import Dict, List, Optional

def process_results(
    results: Dict[str, Any],
    filter_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """处理结果数据"""
    pass
```

## 配置管理

### 1. 使用配置文件

将实验配置放在 JSON 文件中：

```json
{
    "num_agents": 50,
    "num_rounds": 10,
    "use_tot": true,
    "llm_config": {
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
}
```

### 2. 环境变量

敏感信息使用环境变量：

```python
import os
from casevo import OpenAILLM

llm = OpenAILLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
```

### 3. 配置验证

验证配置参数的有效性：

```python
def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置参数"""
    required_keys = ["num_agents", "num_rounds"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if config["num_agents"] <= 0:
        raise ValueError("num_agents must be positive")
    
    return True
```

## 错误处理

### 1. 异常处理

适当处理异常：

```python
try:
    response = llm.send_message(prompt)
except Exception as e:
    logger.error(f"LLM request failed: {e}")
    # 使用默认值或重试
    response = default_response
```

### 2. 日志记录

使用日志记录重要事件：

```python
import logging

logger = logging.getLogger(__name__)

def run_experiment(config):
    logger.info(f"Starting experiment with config: {config}")
    try:
        result = execute_experiment(config)
        logger.info("Experiment completed successfully")
        return result
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise
```

## 性能优化

### 1. 缓存 LLM 请求

使用 `RequestCache` 避免重复请求：

```python
from casevo import RequestCache

cache = RequestCache()
cached_response = cache.get_or_call(
    key=prompt_hash,
    call_func=lambda: llm.send_message(prompt)
)
```

### 2. 批量处理

批量处理相似操作：

```python
# 批量生成嵌入
embeddings = llm.send_embedding([
    text1, text2, text3
])
```

### 3. 异步处理

使用异步处理提高并发性能：

```python
from casevo import ThreadSend

# 异步发送请求
thread_send = ThreadSend(llm)
future = thread_send.send_message_async(prompt)
response = future.get()
```

## 测试

### 1. 单元测试

为关键功能编写单元测试：

```python
import unittest
from casevo import AgentBase, ModelBase

class TestMyAgent(unittest.TestCase):
    def setUp(self):
        # 设置测试环境
        pass
    
    def test_agent_initialization(self):
        # 测试智能体初始化
        pass
```

### 2. 集成测试

测试模块间的集成：

```python
def test_experiment_flow():
    """测试完整实验流程"""
    config = load_test_config()
    result = run_experiment(config)
    assert result is not None
    assert "metrics" in result
```

### 3. 模拟测试

使用模拟对象避免实际 API 调用：

```python
from unittest.mock import Mock

mock_llm = Mock()
mock_llm.send_message.return_value = "Mock response"
```

## 结果管理

### 1. 统一结果目录

所有结果保存到 `experiments/results/`：

```python
import os
from datetime import datetime

results_dir = "experiments/results"
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(results_dir, f"result_{timestamp}.json")
```

### 2. 结果格式

使用一致的 JSON 格式：

```python
result = {
    "experiment_name": "election_simulation",
    "timestamp": timestamp,
    "config": config,
    "metrics": {
        "accuracy": 0.85,
        "consistency": 0.92
    },
    "raw_data": [...]
}
```

### 3. 版本控制

结果文件包含版本信息：

```python
result = {
    "version": "0.4.0",
    "framework_version": casevo.__version__,
    "experiment_version": "1.0",
    ...
}
```

## 调试技巧

### 1. 使用日志

启用详细日志：

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. 检查点

在关键位置添加检查点：

```python
def step(self):
    logger.debug(f"Agent {self.unique_id} step start")
    # ... 执行逻辑
    logger.debug(f"Agent {self.unique_id} step end")
```

### 3. 可视化

使用可视化工具调试：

```python
import matplotlib.pyplot as plt

def visualize_network(model):
    """可视化网络结构"""
    nx.draw(model.grid.G, with_labels=True)
    plt.show()
```

## 代码审查清单

- [ ] 代码遵循命名规范
- [ ] 有适当的文档字符串
- [ ] 使用类型提示
- [ ] 有错误处理
- [ ] 有日志记录
- [ ] 配置可外部化
- [ ] 有单元测试
- [ ] 性能考虑合理
- [ ] 结果格式一致
- [ ] 代码可读性好

## 常见陷阱

1. **忘记设置随机种子**: 导致结果不可重现
2. **硬编码 API Key**: 安全风险
3. **不处理异常**: 程序崩溃
4. **内存泄漏**: 未清理资源
5. **过度调用 LLM**: 成本高、速度慢

## 参考资源

- [Python 代码规范 (PEP 8)](https://pep8.org/)
- [类型提示 (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [Casevo API 文档](../api/README.md)

