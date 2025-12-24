"""
树状思维模块 (Tree of Thought)

实现多路径探索、评估剪枝和回溯机制。
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
import copy

from casevo.core.component import BaseAgentComponent
from casevo.reasoning.chain import BaseStep


class SearchStrategy(Enum):
    """搜索策略枚举"""
    BFS = "breadth_first"
    DFS = "depth_first"
    BEAM = "beam_search"
    BEST_FIRST = "best_first"


@dataclass
class ToTNode:
    """思维树节点"""
    node_id: int
    state: Dict[str, Any]
    score: float = 0.0
    depth: int = 0
    parent: Optional['ToTNode'] = None
    children: List['ToTNode'] = field(default_factory=list)
    is_terminal: bool = False
    is_pruned: bool = False
    reasoning_path: str = ""
    
    def __lt__(self, other: 'ToTNode') -> bool:
        return self.score > other.score
    
    def get_path_to_root(self) -> List['ToTNode']:
        """获取从当前节点到根节点的路径"""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_full_reasoning(self) -> str:
        """获取完整的推理路径"""
        path = self.get_path_to_root()
        return " -> ".join([node.reasoning_path for node in path if node.reasoning_path])


class ToTStep(BaseStep):
    """Tree of Thought 步骤"""
    
    def __init__(self, step_id: str, tar_prompt, num_branches: int = 3):
        super().__init__(step_id, tar_prompt)
        self.num_branches = num_branches
    
    def generate_branches(self, input_state: Dict[str, Any], 
                         agent=None, model=None) -> List[Dict[str, Any]]:
        """生成多个推理分支"""
        branches = []
        
        branch_input = copy.deepcopy(input_state)
        branch_input['num_branches'] = self.num_branches
        branch_input['request_type'] = 'generate_branches'
        
        for i in range(self.num_branches):
            try:
                cur_input = self.pre_process(branch_input, agent, model)
                cur_input['branch_index'] = i
                
                response = self.action(cur_input, agent, model)
                cur_output = self.after_process(cur_input, response, agent, model)
                cur_output['branch_id'] = i
                branches.append(cur_output)
                
            except Exception as e:
                print(f"生成分支 {i} 失败: {e}")
                continue
        
        return branches


class EvaluatorStep(BaseStep):
    """评估步骤"""
    
    def __init__(self, step_id: str, tar_prompt, score_range: tuple = (0.0, 1.0)):
        super().__init__(step_id, tar_prompt)
        self.score_range = score_range
    
    def evaluate(self, state: Dict[str, Any], agent=None, model=None) -> float:
        """评估状态并返回分数"""
        try:
            eval_input = {
                'state': state,
                'request_type': 'evaluate',
                'score_range': self.score_range
            }
            
            cur_input = self.pre_process(eval_input, agent, model)
            response = self.action(cur_input, agent, model)
            result = self.after_process(cur_input, response, agent, model)
            
            score = self._extract_score(result.get('last_response', ''))
            return max(self.score_range[0], min(self.score_range[1], score))
            
        except Exception as e:
            print(f"评估失败: {e}")
            return self.score_range[0]
    
    def _extract_score(self, response: str) -> float:
        """从响应中提取分数"""
        import re
        patterns = [
            r'score[:\s]*([0-9]*\.?[0-9]+)',
            r'评分[:\s]*([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*/\s*[0-9]+',
            r'([0-9]*\.?[0-9]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return self.score_range[0]


class TreeOfThought(BaseAgentComponent):
    """树状思维主类"""
    
    def __init__(self, agent, thought_step: ToTStep, 
                 evaluator_step: Optional[EvaluatorStep] = None,
                 max_depth: int = 5,
                 beam_width: int = 3,
                 pruning_threshold: float = 0.3,
                 search_strategy: SearchStrategy = SearchStrategy.BEAM):
        super().__init__(agent.component_id + "_tot", 'tree_of_thought', agent)
        
        self.thought_step = thought_step
        self.evaluator_step = evaluator_step
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.pruning_threshold = pruning_threshold
        self.search_strategy = search_strategy
        
        self.root: Optional[ToTNode] = None
        self.node_counter = 0
        self.best_node: Optional[ToTNode] = None
        self.all_nodes: List[ToTNode] = []
        self.status = 'init'
    
    def _create_node(self, state: Dict[str, Any], 
                    parent: Optional[ToTNode] = None,
                    reasoning: str = "") -> ToTNode:
        """创建新节点"""
        node = ToTNode(
            node_id=self.node_counter,
            state=state,
            depth=parent.depth + 1 if parent else 0,
            parent=parent,
            reasoning_path=reasoning
        )
        self.node_counter += 1
        self.all_nodes.append(node)
        
        if parent:
            parent.children.append(node)
        
        return node
    
    def _evaluate_node(self, node: ToTNode) -> float:
        """评估节点"""
        if self.evaluator_step:
            score = self.evaluator_step.evaluate(
                node.state, self.agent, self.agent.model
            )
        else:
            score = 1.0 - (node.depth / self.max_depth) * 0.5
        
        node.score = score
        return score
    
    def _should_prune(self, node: ToTNode) -> bool:
        """判断是否应该剪枝"""
        return node.score < self.pruning_threshold
    
    def _is_terminal(self, node: ToTNode) -> bool:
        """判断是否为终止节点"""
        if node.depth >= self.max_depth:
            return True
        if node.state.get('is_final', False):
            return True
        return False
    
    def _expand_node(self, node: ToTNode) -> List[ToTNode]:
        """扩展节点"""
        if node.is_pruned or node.is_terminal:
            return []
        
        branches = self.thought_step.generate_branches(
            node.state, self.agent, self.agent.model
        )
        
        child_nodes = []
        for branch in branches:
            reasoning = branch.get('last_response', str(branch))
            child = self._create_node(branch, node, reasoning)
            
            self._evaluate_node(child)
            
            if self._should_prune(child):
                child.is_pruned = True
            else:
                child_nodes.append(child)
            
            if self._is_terminal(child):
                child.is_terminal = True
        
        return child_nodes
    
    def _beam_search(self, initial_state: Dict[str, Any]) -> ToTNode:
        """束搜索实现"""
        self.root = self._create_node(initial_state, reasoning="初始状态")
        self._evaluate_node(self.root)
        
        current_beam = [self.root]
        
        for depth in range(self.max_depth):
            if not current_beam:
                break
            
            all_children = []
            for node in current_beam:
                if not node.is_terminal and not node.is_pruned:
                    children = self._expand_node(node)
                    all_children.extend(children)
            
            if not all_children:
                break
            
            all_children.sort(key=lambda x: x.score, reverse=True)
            current_beam = all_children[:self.beam_width]
            
            for node in current_beam:
                if node.is_terminal:
                    if self.best_node is None or node.score > self.best_node.score:
                        self.best_node = node
        
        if self.best_node is None:
            all_nodes_sorted = sorted(self.all_nodes, key=lambda x: x.score, reverse=True)
            self.best_node = all_nodes_sorted[0] if all_nodes_sorted else self.root
        
        return self.best_node
    
    def _bfs_search(self, initial_state: Dict[str, Any]) -> ToTNode:
        """广度优先搜索实现"""
        from collections import deque
        
        self.root = self._create_node(initial_state, reasoning="初始状态")
        self._evaluate_node(self.root)
        
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            
            if node.is_terminal:
                if self.best_node is None or node.score > self.best_node.score:
                    self.best_node = node
                continue
            
            if node.depth >= self.max_depth or node.is_pruned:
                continue
            
            children = self._expand_node(node)
            queue.extend(children)
        
        if self.best_node is None:
            all_nodes_sorted = sorted(self.all_nodes, key=lambda x: x.score, reverse=True)
            self.best_node = all_nodes_sorted[0] if all_nodes_sorted else self.root
        
        return self.best_node
    
    def _dfs_search(self, initial_state: Dict[str, Any]) -> ToTNode:
        """深度优先搜索实现"""
        self.root = self._create_node(initial_state, reasoning="初始状态")
        self._evaluate_node(self.root)
        
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            
            if node.is_terminal:
                if self.best_node is None or node.score > self.best_node.score:
                    self.best_node = node
                continue
            
            if node.depth >= self.max_depth or node.is_pruned:
                continue
            
            children = self._expand_node(node)
            stack.extend(reversed(children))
        
        if self.best_node is None:
            all_nodes_sorted = sorted(self.all_nodes, key=lambda x: x.score, reverse=True)
            self.best_node = all_nodes_sorted[0] if all_nodes_sorted else self.root
        
        return self.best_node
    
    def _best_first_search(self, initial_state: Dict[str, Any]) -> ToTNode:
        """最佳优先搜索实现"""
        self.root = self._create_node(initial_state, reasoning="初始状态")
        self._evaluate_node(self.root)
        
        priority_queue = [self.root]
        heapq.heapify(priority_queue)
        
        while priority_queue:
            node = heapq.heappop(priority_queue)
            
            if node.is_terminal:
                if self.best_node is None or node.score > self.best_node.score:
                    self.best_node = node
                continue
            
            if node.depth >= self.max_depth or node.is_pruned:
                continue
            
            children = self._expand_node(node)
            for child in children:
                heapq.heappush(priority_queue, child)
        
        if self.best_node is None:
            all_nodes_sorted = sorted(self.all_nodes, key=lambda x: x.score, reverse=True)
            self.best_node = all_nodes_sorted[0] if all_nodes_sorted else self.root
        
        return self.best_node
    
    def set_input(self, initial_state: Dict[str, Any]):
        """设置初始输入状态"""
        if self.status not in ['init', 'finish']:
            raise Exception("无法设置输入：当前状态不允许")
        
        self.initial_state = initial_state
        self.node_counter = 0
        self.root = None
        self.best_node = None
        self.all_nodes = []
        self.status = 'ready'
    
    def run(self) -> ToTNode:
        """执行树状思维推理"""
        if self.status != 'ready':
            raise Exception("无法运行：状态未就绪")
        
        self.status = 'running'
        
        search_methods = {
            SearchStrategy.BFS: self._bfs_search,
            SearchStrategy.DFS: self._dfs_search,
            SearchStrategy.BEAM: self._beam_search,
            SearchStrategy.BEST_FIRST: self._best_first_search
        }
        
        search_method = search_methods.get(self.search_strategy, self._beam_search)
        result = search_method(self.initial_state)
        
        self.status = 'finish'
        return result
    
    def get_output(self) -> Dict[str, Any]:
        """获取输出结果"""
        if self.status != 'finish':
            raise Exception("无法获取输出：推理未完成")
        
        return {
            'best_node': self.best_node,
            'best_state': self.best_node.state if self.best_node else None,
            'best_score': self.best_node.score if self.best_node else 0,
            'reasoning_path': self.best_node.get_full_reasoning() if self.best_node else "",
            'total_nodes_explored': len(self.all_nodes),
            'max_depth_reached': max(n.depth for n in self.all_nodes) if self.all_nodes else 0
        }
    
    def get_all_paths(self) -> List[List[ToTNode]]:
        """获取所有从根到叶的路径"""
        paths = []
        
        def collect_paths(node: ToTNode, current_path: List[ToTNode]):
            current_path.append(node)
            
            if not node.children or node.is_terminal:
                paths.append(list(current_path))
            else:
                for child in node.children:
                    collect_paths(child, current_path)
            
            current_path.pop()
        
        if self.root:
            collect_paths(self.root, [])
        
        return paths
    
    def backtrack_to(self, node: ToTNode) -> bool:
        """回溯到指定节点并重新探索"""
        if node not in self.all_nodes:
            return False
        
        def invalidate_subtree(n: ToTNode):
            n.is_pruned = True
            for child in n.children:
                invalidate_subtree(child)
        
        for child in node.children:
            invalidate_subtree(child)
        
        node.children = []
        node.is_terminal = False
        node.is_pruned = False
        
        return True


class AdaptiveToT(TreeOfThought):
    """自适应树状思维"""
    
    def __init__(self, agent, thought_step: ToTStep,
                 evaluator_step: Optional[EvaluatorStep] = None,
                 complexity_estimator: Optional[Callable] = None):
        super().__init__(agent, thought_step, evaluator_step)
        self.complexity_estimator = complexity_estimator or self._default_complexity
    
    def _default_complexity(self, state: Dict[str, Any]) -> float:
        """默认复杂度估计"""
        content = str(state)
        return min(1.0, len(content) / 1000)
    
    def set_input(self, initial_state: Dict[str, Any]):
        """根据复杂度自适应调整参数"""
        complexity = self.complexity_estimator(initial_state)
        
        if complexity < 0.3:
            self.max_depth = 3
            self.beam_width = 2
            self.search_strategy = SearchStrategy.DFS
        elif complexity < 0.7:
            self.max_depth = 5
            self.beam_width = 3
            self.search_strategy = SearchStrategy.BEAM
        else:
            self.max_depth = 7
            self.beam_width = 5
            self.search_strategy = SearchStrategy.BEST_FIRST
        
        super().set_input(initial_state)


class ToTChainPool:
    """ToT 链池"""
    
    def __init__(self, thread_num: int = 4):
        import queue
        import threading
        
        self.status = 'init'
        self.tot_queue = queue.Queue()
        self.thread_num = thread_num
        self.results = {}
        self.lock = threading.Lock()
    
    def add_tots(self, tots: List[TreeOfThought]):
        """添加多个 ToT 实例到队列"""
        for tot in tots:
            self.tot_queue.put(tot)
        self.status = 'ready'
    
    def _worker(self):
        """工作线程"""
        import time
        
        while True:
            try:
                tot = self.tot_queue.get(timeout=1)
                result = tot.run()
                
                with self.lock:
                    self.results[tot.component_id] = tot.get_output()
                
                self.tot_queue.task_done()
            except:
                break
            
            time.sleep(0.1)
    
    def start_pool(self) -> Dict[str, Dict[str, Any]]:
        """启动线程池并等待完成"""
        import threading
        
        if self.status != 'ready':
            raise Exception("线程池状态未就绪")
        
        threads = []
        for _ in range(self.thread_num):
            t = threading.Thread(target=self._worker)
            t.start()
            threads.append(t)
        
        self.tot_queue.join()
        
        for t in threads:
            t.join(timeout=1)
        
        return self.results

