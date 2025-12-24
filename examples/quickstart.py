"""
Casevo 快速开始示例

展示如何使用 Casevo 进行简单的多智能体模拟。
"""

import networkx as nx


def create_simple_simulation():
    """创建一个简单的模拟示例"""
    
    # 导入 Casevo 模块
    from casevo import (
        AgentBase,
        ModelBase,
        create_default_llm,
        ThoughtChain,
        BaseStep,
    )
    
    print("=" * 60)
    print("Casevo 快速开始示例")
    print("=" * 60)
    
    # 1. 创建一个简单的网络结构
    print("\n1. 创建网络结构...")
    G = nx.complete_graph(3)  # 3 个完全连接的节点
    print(f"   创建了 {G.number_of_nodes()} 个节点的完全图")
    
    # 2. 创建 LLM 实例（可选 - 如果没有 API key 可以跳过）
    print("\n2. 初始化 LLM...")
    try:
        llm = create_default_llm()
        print("   LLM 初始化成功")
    except Exception as e:
        print(f"   LLM 初始化失败: {e}")
        print("   继续运行演示（不使用真实 LLM）...")
        llm = None
    
    # 3. 展示框架结构
    print("\n3. Casevo 模块结构:")
    print("   - casevo.core: 核心模块 (AgentBase, ModelBase)")
    print("   - casevo.llm: LLM 接口")
    print("   - casevo.memory: 记忆系统")
    print("   - casevo.reasoning: 推理模块 (CoT, ToT)")
    print("   - casevo.utils: 工具模块")
    
    # 4. 展示导入的类
    print("\n4. 可用的主要类:")
    from casevo import (
        # 核心
        AgentBase, ModelBase,
        # LLM
        LLM_INTERFACE, OpenAILLM,
        # 记忆
        Memory, AdvancedMemory,
        # 推理
        ThoughtChain, TreeOfThought, CollaborativeDecisionMaker,
    )
    
    classes = [
        ("AgentBase", "智能体基类"),
        ("ModelBase", "模型基类"),
        ("LLM_INTERFACE", "LLM 接口"),
        ("Memory", "基础记忆"),
        ("AdvancedMemory", "高级记忆"),
        ("ThoughtChain", "思维链 (CoT)"),
        ("TreeOfThought", "树状思维 (ToT)"),
        ("CollaborativeDecisionMaker", "协同决策"),
    ]
    
    for cls_name, desc in classes:
        print(f"   - {cls_name}: {desc}")
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("更多详细示例请参考 experiments/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    create_simple_simulation()

