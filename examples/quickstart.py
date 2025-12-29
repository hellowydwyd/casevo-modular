"""
Casevo 快速开始示例

展示如何使用 Casevo 进行简单的多智能体模拟。
本示例使用 Mock LLM，无需配置真实 API 即可运行。
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import networkx as nx

from casevo import (
    AgentBase,
    ModelBase,
    ThoughtChain,
    BaseStep,
    ChoiceStep,
    LLM_INTERFACE,
)


# ============================================================
# Mock LLM（用于演示，无需真实 API）
# ============================================================
class MockLLM(LLM_INTERFACE):
    """
    模拟 LLM，用于演示和测试。
    返回预设的响应，无需配置真实 API。
    """
    
    def __init__(self):
        self._embedding_function = MockEmbeddingFunction()
    
    def send_message(self, prompt, json_flag=False):
        """模拟发送消息"""
        # 简单的模拟响应逻辑
        if "选择" in prompt or "choose" in prompt.lower():
            return "经过深思熟虑，我选择 A 选项。"
        elif "评分" in prompt or "score" in prompt.lower():
            return "综合评估后，我给出的评分是 8.5 分。"
        else:
            return "这是一个模拟的 LLM 响应。我理解了您的问题，并给出了回答。"
    
    def send_embedding(self, text_list):
        """返回模拟的嵌入向量"""
        return [[0.1] * 1536 for _ in text_list]
    
    def get_lang_embedding(self):
        """返回嵌入函数"""
        return self._embedding_function


class MockEmbeddingFunction:
    """Mock 嵌入函数，兼容 ChromaDB"""
    
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return [[0.1] * 1536 for _ in input]
    
    def name(self):
        return "mock_embedding_function"
    
    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return [[0.1] * 1536 for _ in input]
    
    def is_legacy(self):
        return False
    
    def default_space(self):
        return "cosine"
    
    def supported_spaces(self):
        return ["cosine", "l2", "ip"]


# ============================================================
# 自定义智能体
# ============================================================
class SimpleAgent(AgentBase):
    """
    简单的智能体实现示例。
    
    每个智能体有自己的描述、记忆和思维链。
    """
    
    def __init__(self, unique_id, model, description, context=None):
        super().__init__(unique_id, model, description, context)
        self.interaction_count = 0
        self.opinions = []
    
    def step(self):
        """
        智能体每一步的行为：
        1. 获取邻居信息
        2. 记录交互到记忆
        3. 形成观点
        """
        self.interaction_count += 1
        
        # 获取邻居
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbor_ids = [n.unique_id for n in neighbors]
        
        # 添加记忆：记录本次交互
        self.memory.add_short_memory(
            source=self.component_id,
            target=f"neighbors_{neighbor_ids}",
            action="observe",
            content=f"第 {self.interaction_count} 轮：观察到 {len(neighbors)} 个邻居"
        )
        
        # 简单的观点形成逻辑
        opinion = f"智能体 {self.unique_id} 在第 {self.interaction_count} 轮与 {len(neighbors)} 个邻居交互"
        self.opinions.append(opinion)
        
        return opinion


# ============================================================
# 自定义模型
# ============================================================
class SimpleModel(ModelBase):
    """
    简单的模型实现示例。
    
    管理多个智能体，并模拟它们之间的交互。
    """
    
    def __init__(self, graph, llm, num_agents=5):
        # 创建临时的 prompt 目录
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        prompt_dir = os.path.join(self.temp_dir, 'prompts')
        os.makedirs(prompt_dir)
        
        # 创建必要的 prompt 模板
        with open(os.path.join(prompt_dir, 'reflect.txt'), 'w', encoding='utf-8') as f:
            f.write("请根据以下记忆进行反思：\n长期记忆：{{ extra.long_memory }}\n短期记忆：{{ extra.short_memory }}")
        
        super().__init__(
            tar_graph=graph,
            llm=llm,
            prompt_path=prompt_dir,
            memory_num=5
        )
        
        self.num_agents = num_agents
        self.round = 0
        
        # 创建智能体
        self._create_agents()
    
    def _create_agents(self):
        """创建智能体"""
        descriptions = [
            {"role": "分析师", "trait": "理性、逻辑思维强"},
            {"role": "创新者", "trait": "创造力强、喜欢新想法"},
            {"role": "协调者", "trait": "善于沟通、重视团队"},
            {"role": "执行者", "trait": "务实、注重效率"},
            {"role": "观察者", "trait": "细心、善于发现问题"},
        ]
        
        for i in range(min(self.num_agents, len(descriptions))):
            agent = SimpleAgent(
                unique_id=i,
                model=self,
                description=descriptions[i],
                context={"round": 0}
            )
            self.add_agent(agent, i)
    
    def step(self):
        """执行一轮模拟"""
        self.round += 1
        print(f"\n--- 第 {self.round} 轮 ---")
        
        # 更新所有智能体的上下文
        for agent in self.agent_list:
            agent.context = {"round": self.round}
        
        # 执行调度
        self.schedule.step()
        
        # 打印状态
        for agent in self.agent_list:
            if agent.opinions:
                print(f"  {agent.opinions[-1]}")
        
        return self.round
    
    def run(self, rounds=3):
        """运行多轮模拟"""
        print(f"\n开始模拟：{self.num_agents} 个智能体，{rounds} 轮")
        print("=" * 50)
        
        for _ in range(rounds):
            self.step()
        
        print("\n" + "=" * 50)
        print("模拟结束")
        
        return self.get_results()
    
    def get_results(self):
        """获取模拟结果"""
        return {
            "total_rounds": self.round,
            "agents": [
                {
                    "id": a.unique_id,
                    "description": a.description,
                    "interactions": a.interaction_count,
                    "opinions": a.opinions
                }
                for a in self.agent_list
            ]
        }
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# ============================================================
# 思维链示例
# ============================================================
def demonstrate_thought_chain():
    """演示思维链 (Chain of Thought) 的使用"""
    print("\n" + "=" * 50)
    print("思维链 (CoT) 演示")
    print("=" * 50)
    
    # 创建 Mock 环境
    llm = MockLLM()
    graph = nx.complete_graph(3)
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    prompt_dir = os.path.join(temp_dir, 'prompts')
    os.makedirs(prompt_dir)
    
    # 创建思维链的 prompt 模板
    with open(os.path.join(prompt_dir, 'reflect.txt'), 'w', encoding='utf-8') as f:
        f.write("反思：{{ extra }}")
    with open(os.path.join(prompt_dir, 'think.txt'), 'w', encoding='utf-8') as f:
        f.write("请思考这个问题：{{ extra.question }}")
    with open(os.path.join(prompt_dir, 'choose.txt'), 'w', encoding='utf-8') as f:
        f.write("请从以下选项中选择：{{ extra.options }}")
    
    # 创建模型和智能体
    model = ModelBase(graph, llm, prompt_path=prompt_dir)
    
    class ThinkingAgent(AgentBase):
        def step(self):
            pass
    
    agent = ThinkingAgent(0, model, {"role": "思考者"}, None)
    model.add_agent(agent, 0)
    
    # 创建思维链步骤
    think_prompt = model.prompt_factory.get_template('think.txt')
    choose_prompt = model.prompt_factory.get_template('choose.txt')
    
    step1 = BaseStep("think", think_prompt)
    step2 = ChoiceStep("choose", choose_prompt)
    
    # 设置思维链
    agent.setup_chain({
        'decision': [step1, step2]
    })
    
    # 执行思维链
    print("\n执行决策思维链...")
    chain = agent.chains['decision']
    chain.set_input({
        'question': '我们应该采用什么策略？',
        'options': 'A. 保守策略  B. 激进策略  C. 平衡策略'
    })
    
    try:
        chain.run_step()
        output = chain.get_output()
        print(f"思维链输出: {output}")
        print(f"最终选择: {output.get('choice', 'N/A')}")
    except Exception as e:
        print(f"思维链执行出错: {e}")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n思维链演示完成！")


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数：运行完整的演示"""
    print("=" * 60)
    print("Casevo 快速开始示例")
    print("=" * 60)
    
    print("\n本示例展示 Casevo 框架的核心功能：")
    print("1. 创建多智能体模型")
    print("2. 智能体交互与记忆")
    print("3. 思维链 (Chain of Thought)")
    
    # 1. 创建并运行简单模拟
    print("\n" + "=" * 50)
    print("多智能体模拟演示")
    print("=" * 50)
    
    # 创建网络结构
    graph = nx.watts_strogatz_graph(5, 2, 0.3)  # 小世界网络
    print(f"\n创建小世界网络：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    
    # 创建 Mock LLM
    llm = MockLLM()
    print("使用 Mock LLM（无需真实 API）")
    
    # 创建并运行模型
    model = SimpleModel(graph, llm, num_agents=5)
    results = model.run(rounds=3)
    
    # 打印结果
    print("\n模拟结果:")
    for agent_info in results['agents']:
        print(f"  智能体 {agent_info['id']} ({agent_info['description']['role']}): "
              f"{agent_info['interactions']} 次交互")
    
    # 清理
    model.cleanup()
    
    # 2. 思维链演示
    demonstrate_thought_chain()
    
    # 3. 总结
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n下一步：")
    print("  - 查看 experiments/ 目录了解更多实验场景")
    print("  - 阅读 docs/guides/getting_started.md 了解详细使用方法")
    print("  - 配置真实 LLM API 开始您的模拟实验")


if __name__ == "__main__":
    main()
