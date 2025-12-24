"""
集成真实 LLM 的选举投票实验

使用 OpenAI API 进行智能体决策。
注意：此实验会消耗 API 额度，建议先用少量智能体测试。
"""

import mesa
import networkx as nx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random
import json
import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from casevo import OpenAILLM, create_default_llm, create_llm
from casevo.prompt import PromptFactory


class VoteChoice(Enum):
    """投票选择"""
    BIDEN = "Biden"
    TRUMP = "Trump"
    UNDECIDED = "Undecided"


class VoterCategory(Enum):
    """选民类别（基于 Pew Research 政治类型学 - 论文原始设置）"""
    CONSERVATIVE_WHITE_MALE = "Conservative White Male"
    FINANCIALLY_STABLE_EDUCATED = "Financially Stable Educated White Male"
    LOW_INCOME_RURAL_FEMALE = "Low-Income Rural White Female"
    YOUNG_DIVERSE = "Young Diverse Group"
    STRUGGLING_WHITE_FEMALE = "Low-Income White Female"
    STRUGGLING_YOUNG_LIBERAL = "Financially Struggling Young Liberal"
    ELDERLY_RELIGIOUS_FEMALE = "Elderly Religious Female"
    HIGH_INCOME_EDUCATED = "High-Income Highly Educated White"
    EDUCATED_LIBERAL_YOUNG = "Highly Educated Liberal Young White"


# 详细选民配置（按照原论文标准）
VOTER_PROFILES = {
    VoterCategory.CONSERVATIVE_WHITE_MALE: {
        'distribution': 0.12,
        'initial_vote': VoteChoice.TRUMP,
        'template': {
            'age_range': (50, 65),
            'gender': '男性',
            'race': '白人',
            'location': '农村社区',
            'education': '高中或部分大学',
            'income': '中等收入',
            'religion': '虔诚基督徒',
        },
        'background': '''作为一名{age}岁的保守派白人男性，居住在美国{state}州的农村社区。
作为虔诚的基督徒，他认为宗教在公共生活中至关重要，支持政府政策应维护宗教价值观。
他积极参与地方事务，主张减少政府对社会的干预。在堕胎和同性婚姻问题上持保守观点。
他认为当代社会对白人存在逆向歧视。支持强大的国家军事力量，认为军事实力而非外交才是确保和平的最佳手段。''',
        'issues': ['宗教自由', '减少政府干预', '传统家庭价值', '强大军事', '边境安全'],
    },
    
    VoterCategory.FINANCIALLY_STABLE_EDUCATED: {
        'distribution': 0.11,
        'initial_vote': VoteChoice.TRUMP,
        'template': {
            'age_range': (50, 70),
            'gender': '男性',
            'race': '白人',
            'location': '郊区',
            'education': '大学学历',
            'income': '高收入',
            'religion': '新教徒',
        },
        'background': '''作为一名{age}岁的白人男性，居住在{state}州郊区的高档社区。
拥有大学学历，在金融/商业领域工作多年，财务状况稳定。
他支持减税政策和自由市场经济，认为政府监管过多会阻碍经济发展。
关注股市表现和退休金安全。他认可特朗普的经济政策，但在某些社会议题上态度较温和。
重视法律与秩序，担忧城市犯罪率上升。''',
        'issues': ['减税', '股市经济', '法律秩序', '退休保障', '减少监管'],
    },
    
    VoterCategory.LOW_INCOME_RURAL_FEMALE: {
        'distribution': 0.10,
        'initial_vote': VoteChoice.UNDECIDED,
        'template': {
            'age_range': (35, 55),
            'gender': '女性',
            'race': '白人',
            'location': '农村',
            'education': '高中学历',
            'income': '低收入',
            'religion': '基督徒',
        },
        'background': '''作为一名{age}岁的白人女性，居住在{state}州的农村地区。
高中学历，从事服务业或零售业工作，收入有限。
她担心医疗费用和子女教育问题，希望政府能提供更多帮助。
在社会议题上较为保守，但在经济政策上希望获得更多支持。
她对政治不太感兴趣，主要关心日常生活开支和家庭安全。''',
        'issues': ['医疗费用', '就业机会', '子女教育', '生活成本', '社区安全'],
    },
    
    VoterCategory.YOUNG_DIVERSE: {
        'distribution': 0.12,
        'initial_vote': VoteChoice.BIDEN,
        'template': {
            'age_range': (22, 35),
            'gender': '随机',
            'race': '多元化',
            'location': '城市',
            'education': '大学在读或毕业',
            'income': '中低收入',
            'religion': '无特定宗教或世俗',
        },
        'background': '''作为一名{age}岁的年轻{race}，居住在{state}州的城市地区。
大学学历，关注社会公正和环境问题。
强烈支持种族平等运动，认为系统性种族主义是真实存在的问题。
关注气候变化，支持绿色新政。希望大学学费能够降低或免费。
对传统政治持怀疑态度，但认为投票参与很重要。支持LGBTQ+权利。''',
        'issues': ['种族正义', '气候变化', '大学学费', '社会公正', 'LGBTQ权利'],
    },
    
    VoterCategory.STRUGGLING_WHITE_FEMALE: {
        'distribution': 0.10,
        'initial_vote': VoteChoice.UNDECIDED,
        'template': {
            'age_range': (30, 50),
            'gender': '女性',
            'race': '白人',
            'location': '小城镇',
            'education': '高中学历',
            'income': '低收入',
            'religion': '基督徒',
        },
        'background': '''作为一名{age}岁的白人女性，居住在{state}州的小城镇。
高中学历，可能是单亲妈妈或家庭经济支柱。
最关心的是能否支付账单和保住工作。医疗保险是一个大问题。
她对两党都不太信任，觉得政客们不关心普通人的生活。
希望能有更多工作机会和更低的医疗费用。''',
        'issues': ['就业', '医疗保险', '生活成本', '子女照顾', '工资水平'],
    },
    
    VoterCategory.STRUGGLING_YOUNG_LIBERAL: {
        'distribution': 0.11,
        'initial_vote': VoteChoice.BIDEN,
        'template': {
            'age_range': (25, 40),
            'gender': '随机',
            'race': '白人或混血',
            'location': '城市',
            'education': '大学学历',
            'income': '中低收入',
            'religion': '无宗教信仰',
        },
        'background': '''作为一名{age}岁的年轻人，居住在{state}州的城市地区。
拥有大学学历，但背负学生贷款，经济状况不稳定。
从事创意产业或服务业，收入不高但有社会责任感。
强烈支持进步议程，包括全民医保、提高最低工资、应对气候变化。
对资本主义持批评态度，希望有更公平的经济体系。''',
        'issues': ['学生贷款', '全民医保', '最低工资', '气候变化', '经济不平等'],
    },
    
    VoterCategory.ELDERLY_RELIGIOUS_FEMALE: {
        'distribution': 0.10,
        'initial_vote': VoteChoice.TRUMP,
        'template': {
            'age_range': (65, 80),
            'gender': '女性',
            'race': '白人',
            'location': '郊区或农村',
            'education': '高中学历',
            'income': '固定收入（退休金）',
            'religion': '虔诚基督徒',
        },
        'background': '''作为一名{age}岁的退休女性，居住在{state}州。
虔诚的基督徒，每周参加教会活动。信仰是生活的核心。
强烈反对堕胎，认为这是道德问题。支持传统婚姻定义。
关心社会保障和医疗保险（Medicare）的稳定。
担忧国家道德衰落，希望恢复传统价值观。对移民问题持保守态度。''',
        'issues': ['反对堕胎', '宗教自由', '社会保障', 'Medicare', '传统价值'],
    },
    
    VoterCategory.HIGH_INCOME_EDUCATED: {
        'distribution': 0.12,
        'initial_vote': VoteChoice.UNDECIDED,
        'template': {
            'age_range': (40, 60),
            'gender': '随机',
            'race': '白人',
            'location': '郊区高档社区',
            'education': '研究生学历',
            'income': '高收入',
            'religion': '主流新教或无宗教',
        },
        'background': '''作为一名{age}岁的专业人士，居住在{state}州的高档郊区。
拥有研究生学历，在医疗、法律或科技行业工作。
经济上倾向保守（支持减税），但社会议题上较为自由。
关注教育质量和社区安全。对特朗普的言行风格感到不适。
希望有更文明的政治氛围，重视专业性和能力。''',
        'issues': ['减税', '教育质量', '专业治理', '社会稳定', '国际关系'],
    },
    
    VoterCategory.EDUCATED_LIBERAL_YOUNG: {
        'distribution': 0.12,
        'initial_vote': VoteChoice.BIDEN,
        'template': {
            'age_range': (28, 45),
            'gender': '随机',
            'race': '白人',
            'location': '城市',
            'education': '研究生学历',
            'income': '中高收入',
            'religion': '无宗教或精神但非宗教',
        },
        'background': '''作为一名{age}岁的高学历专业人士，居住在{state}州的城市地区。
在学术界、媒体、科技或非营利组织工作。
强烈关注社会公正议题，支持BLM运动和移民权利。
认为气候变化是紧迫危机，支持激进的环保政策。
对特朗普持强烈反对态度，认为他威胁民主制度。
积极参与政治活动，会说服他人投票。''',
        'issues': ['社会公正', '气候危机', '民主制度', '移民权利', '科学政策'],
    },
}

# 美国州列表（用于生成背景）
US_STATES = [
    '德克萨斯', '佛罗里达', '宾夕法尼亚', '俄亥俄', '密歇根', 
    '北卡罗来纳', '亚利桑那', '威斯康星', '佐治亚', '内华达',
    '爱荷华', '明尼苏达', '科罗拉多', '弗吉尼亚', '新罕布什尔'
]

# 种族选项
RACE_OPTIONS = ['白人', '非裔美国人', '拉丁裔', '亚裔', '混血']


@dataclass
class VoterProfile:
    """详细选民画像（按论文标准）"""
    category: VoterCategory           # 选民类别
    name: str                         # 姓名
    age: int                          # 年龄
    gender: str                       # 性别
    race: str                         # 种族
    state: str                        # 所在州
    location_type: str                # 居住地类型
    education: str                    # 教育程度
    income_level: str                 # 收入水平
    religion: str                     # 宗教信仰
    background_story: str             # 详细背景故事
    key_issues: List[str]             # 关注议题
    initial_preference: VoteChoice    # 初始投票倾向
    susceptibility: float             # 易受影响程度 (0-1)


class LLMVoterAgent(mesa.Agent):
    """
    使用 LLM 进行决策的选民智能体
    """
    
    def __init__(self, unique_id: int, model: 'LLMElectionModel', 
                 profile: VoterProfile):
        super().__init__(unique_id, model)
        self.profile = profile
        self.current_vote = profile.initial_preference
        self.vote_history: List[Dict[str, Any]] = []
        self.reasoning_history: List[str] = []
        self.neighbors: List[int] = []
    
    def make_decision(self, debate_info: str, neighbor_opinions: List[str]) -> VoteChoice:
        """
        使用 LLM 进行投票决策
        
        Args:
            debate_info: 辩论信息
            neighbor_opinions: 邻居的观点
        
        Returns:
            投票选择
        """
        # 构建系统提示（确保中立角色扮演）
        system_prompt = """你是一个角色扮演专家。你的任务是完全代入给定角色的身份、价值观和思维方式来做出决定。

关键指令：
1. 你必须完全按照角色的背景、价值观和关注议题来思考
2. 不要加入你自己的观点或偏见
3. 角色的决定应该符合其人口统计特征和政治倾向
4. 如果角色是保守派，就像保守派那样思考；如果是自由派，就像自由派那样思考
5. 每个角色都有自己的"真实"，请尊重这种多样性

这是一个研究模拟，需要真实反映不同政治观点的选民行为。"""

        # 构建用户提示（按论文标准的详细人设）
        user_prompt = f"""请扮演以下美国选民，用第一人称做出投票决定。

## 你的身份
姓名：{self.profile.name}
年龄：{self.profile.age}岁
性别：{self.profile.gender}
种族：{self.profile.race}
居住地：{self.profile.state}州（{self.profile.location_type}）
教育程度：{self.profile.education}
收入水平：{self.profile.income_level}
宗教信仰：{self.profile.religion}

## 你的背景故事
{self.profile.background_story}

## 你最关心的议题（按优先级）
{', '.join(self.profile.key_issues)}

## 你目前的投票倾向
{self.current_vote.value}

---

## 今天的辩论内容
{debate_info}

## 你周围人的看法
{chr(10).join(neighbor_opinions) if neighbor_opinions else '（你还没有和邻居或朋友讨论过这次选举）'}

---

## 决策任务

作为 {self.profile.name}，基于你的具体背景（{self.profile.age}岁{self.profile.location_type}{self.profile.religion}）、你最关心的议题、以及今天辩论中两位候选人的表现，你会如何投票？

注意：
- 重点考虑辩论内容是否触及你最关心的议题
- 考虑哪位候选人的立场更符合你的价值观和利益
- 周围人的意见可能会影响你，但最终由你决定

请按以下格式回答：

【我的考虑】作为一个{self.profile.age}岁的{self.profile.location_type}{self.profile.race}{self.profile.gender}，我主要关心{self.profile.key_issues[0]}...（1-2句话说明你的思考）

【最终决定】Biden / Trump / Undecided（只写一个选项）
"""
        
        # 组合完整提示
        prompt = f"[系统指令]\n{system_prompt}\n\n[用户任务]\n{user_prompt}"
        
        try:
            response = self.model.llm.send_message(prompt)
            
            # 解析响应
            self.reasoning_history.append(response)
            
            # 提取投票决定
            response_lower = response.lower()
            if 'biden' in response_lower and '最终决定' in response:
                if 'biden' in response.split('最终决定')[-1].lower():
                    return VoteChoice.BIDEN
            if 'trump' in response_lower and '最终决定' in response:
                if 'trump' in response.split('最终决定')[-1].lower():
                    return VoteChoice.TRUMP
            
            # 默认保持原选择
            return self.current_vote
            
        except Exception as e:
            print(f"  Agent {self.unique_id} LLM 调用失败: {e}")
            return self.current_vote
    
    def step(self):
        """每轮执行"""
        # 获取辩论信息
        debate = self.model.get_current_debate()
        key_issues = debate.get('key_issues', [])
        debate_info = f"""主题: {debate['topic']}
涉及议题: {', '.join(key_issues)}

Biden 的发言:
{debate['biden']}

Trump 的发言:
{debate['trump']}"""
        
        # 获取邻居意见
        neighbor_opinions = []
        for n_id in self.neighbors[:3]:  # 限制邻居数量以节省 API 调用
            for agent in self.model.schedule.agents:
                if agent.unique_id == n_id:
                    neighbor_opinions.append(f"邻居{n_id}支持: {agent.current_vote.value}")
                    break
        
        # 使用 LLM 决策
        new_vote = self.make_decision(debate_info, neighbor_opinions)
        
        # 记录历史
        self.vote_history.append({
            'round': self.model.current_round,
            'vote': new_vote.value,
            'previous': self.current_vote.value
        })
        
        self.current_vote = new_vote


class LLMElectionModel(mesa.Model):
    """
    使用 LLM 的选举模型
    """
    
    def __init__(self, num_voters: int = 10, llm: OpenAILLM = None):
        """
        初始化模型
        
        Args:
            num_voters: 选民数量（建议先用小数量测试）
            llm: LLM 实例
        """
        super().__init__()
        
        self.num_voters = num_voters
        self.llm = llm or create_default_llm()
        self.schedule = mesa.time.RandomActivation(self)
        self.current_round = 0
        
        # 创建小世界网络
        self.network = nx.watts_strogatz_graph(num_voters, min(4, num_voters-1), 0.3)
        
        # 6轮辩论内容（平衡双方观点，包含各自优势议题）
        self.debates = [
            {
                'topic': '经济与就业',
                'biden': '疫情前的经济繁荣建立在奥巴马政府8年恢复的基础上。我将为中产阶级减税，投资基础设施创造就业。特朗普的减税主要惠及富人和大企业。',
                'trump': '在我上任之前，经济萎靡不振。我的减税政策创造了历史最低失业率，股市屡创新高。我们签署了更好的贸易协议，让制造业回归美国。拜登的增税计划会扼杀经济。',
                'key_issues': ['经济增长', '就业', '税收', '股市']
            },
            {
                'topic': '边境安全与移民',
                'biden': '我们是移民国家，需要人道的移民政策。我支持边境安全，但反对将儿童与父母分离。我们需要为梦想者提供合法身份的途径。',
                'trump': '我建造了边境墙，非法越境大幅减少。拜登会开放边境，让MS-13黑帮和毒品涌入。我签署了《留在墨西哥》政策，保护美国工人的就业机会。',
                'key_issues': ['边境安全', '非法移民', '墙', '就业']
            },
            {
                'topic': '新冠疫情应对',
                'biden': '22万美国人死亡，总统需要承担责任。我们需要科学的防疫政策，全民戴口罩，大规模检测。我会听取专家意见，制定国家计划。',
                'trump': '我在一月就关闭了与中国的边境，专家说救了数百万生命。我们启动了"曲速行动"，疫苗即将问世，创造了奇迹。民主党州长的封锁摧毁了经济和生活。',
                'key_issues': ['疫情防控', '疫苗', '封锁', '公共卫生']
            },
            {
                'topic': '法律与秩序',
                'biden': '我支持执法部门，但也需要改革。我反对"削减警察经费"，但警察暴力必须解决。我们需要建立信任，既保护社区安全也保护公民权利。',
                'trump': '民主党城市犯罪飙升，因为他们不支持警察。BLM和Antifa的骚乱造成数十亿美元损失。我是法律与秩序的总统，我保护郊区社区免受危险。',
                'key_issues': ['犯罪', '警察', '骚乱', '社区安全']
            },
            {
                'topic': '宗教自由与传统价值',
                'biden': '我是虔诚的天主教徒，尊重所有人的信仰自由。但我也相信政教分离。我支持女性选择权，这是法律确立的权利。',
                'trump': '我是最支持宗教自由的总统。我任命了三位保守派大法官，保护未出生的生命。我取消了强迫宗教团体违背信仰的规定。拜登会攻击你们的信仰。',
                'key_issues': ['宗教自由', '堕胎', '最高法院', '传统价值']
            },
            {
                'topic': '中国与国际贸易',
                'biden': '我们需要与盟友合作应对中国。我会重建联盟，联合施压。特朗普的贸易战伤害了美国农民，让纳税人买单。',
                'trump': '中国多年占美国便宜，只有我敢站出来。我的关税政策迫使中国购买更多美国商品，保护了知识产权。拜登和他儿子与中国有生意往来，他会软弱应对。',
                'key_issues': ['中国', '贸易战', '关税', '制造业']
            }
        ]
        
        # 创建选民
        self._create_voters()
        
        # 结果记录
        self.results_history = []
    
    def _create_voters(self):
        """创建选民代理（按论文标准的详细人设）"""
        # 常见美国名字库
        male_first_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 
                           'Thomas', 'Charles', 'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Mark',
                           'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian']
        female_first_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Barbara', 'Elizabeth', 'Susan',
                             'Jessica', 'Sarah', 'Karen', 'Lisa', 'Nancy', 'Betty', 'Margaret', 'Sandra',
                             'Ashley', 'Kimberly', 'Emily', 'Donna', 'Michelle', 'Dorothy', 'Carol', 'Amanda']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                     'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                     'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White']
        
        # 获取选民类型及其分布权重
        categories = list(VOTER_PROFILES.keys())
        weights = [VOTER_PROFILES[c]['distribution'] for c in categories]
        
        for i in range(self.num_voters):
            # 按真实分布随机选择选民类型
            category = random.choices(categories, weights=weights)[0]
            config = VOTER_PROFILES[category]
            template = config['template']
            
            # 生成年龄
            age = random.randint(template['age_range'][0], template['age_range'][1])
            
            # 生成性别
            if template['gender'] == '随机':
                gender = random.choice(['男性', '女性'])
            else:
                gender = template['gender']
            
            # 生成种族
            if template['race'] == '多元化':
                race = random.choice(RACE_OPTIONS)
            elif template['race'] == '白人或混血':
                race = random.choice(['白人', '混血'])
            else:
                race = template['race']
            
            # 生成名字
            if gender == '男性':
                first_name = random.choice(male_first_names)
            else:
                first_name = random.choice(female_first_names)
            name = f"{first_name} {random.choice(last_names)}"
            
            # 选择州
            state = random.choice(US_STATES)
            
            # 生成背景故事
            background = config['background'].format(
                age=age,
                state=state,
                race=race
            )
            
            # 根据选民类型设置易受影响程度
            if category in [VoterCategory.LOW_INCOME_RURAL_FEMALE, 
                           VoterCategory.STRUGGLING_WHITE_FEMALE,
                           VoterCategory.HIGH_INCOME_EDUCATED]:
                susceptibility = random.uniform(0.5, 0.85)  # 高影响 - 摇摆选民
            elif category in [VoterCategory.CONSERVATIVE_WHITE_MALE,
                             VoterCategory.ELDERLY_RELIGIOUS_FEMALE,
                             VoterCategory.EDUCATED_LIBERAL_YOUNG]:
                susceptibility = random.uniform(0.1, 0.35)  # 低影响 - 立场坚定
            else:
                susceptibility = random.uniform(0.35, 0.6)  # 中等影响
            
            profile = VoterProfile(
                category=category,
                name=name,
                age=age,
                gender=gender,
                race=race,
                state=state,
                location_type=template['location'],
                education=template['education'],
                income_level=template['income'],
                religion=template['religion'],
                background_story=background,
                key_issues=config['issues'].copy(),
                initial_preference=config['initial_vote'],
                susceptibility=susceptibility
            )
            
            agent = LLMVoterAgent(i, self, profile)
            self.schedule.add(agent)
            agent.neighbors = list(self.network.neighbors(i))
        
        # 打印选民分布统计
        dist = {}
        vote_dist = {'Biden': 0, 'Trump': 0, 'Undecided': 0}
        for agent in self.schedule.agents:
            cat = agent.profile.category.value
            dist[cat] = dist.get(cat, 0) + 1
            vote_dist[agent.profile.initial_preference.value] += 1
        
        print("\n选民类型分布（按 Pew 研究标准）:")
        for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count} ({count/self.num_voters*100:.1f}%)")
        
        print(f"\n初始投票倾向: Biden={vote_dist['Biden']}, Trump={vote_dist['Trump']}, Undecided={vote_dist['Undecided']}")
    
    def get_current_debate(self) -> Dict[str, str]:
        """获取当前轮次的辩论"""
        idx = min(self.current_round, len(self.debates) - 1)
        return self.debates[idx]
    
    def count_votes(self) -> Dict[str, int]:
        """统计投票"""
        counts = {'Biden': 0, 'Trump': 0, 'Undecided': 0}
        for agent in self.schedule.agents:
            counts[agent.current_vote.value] += 1
        return counts
    
    def step(self):
        """执行一轮"""
        print(f"\n--- 第 {self.current_round + 1} 轮辩论 ---")
        print(f"主题: {self.get_current_debate()['topic']}")
        
        # 执行所有智能体
        for agent in self.schedule.agents:
            print(f"  处理智能体 {agent.unique_id}...")
            agent.step()
            time.sleep(0.5)  # 避免 API 速率限制
        
        # 记录结果
        votes = self.count_votes()
        self.results_history.append({
            'round': self.current_round + 1,
            'votes': votes
        })
        
        print(f"本轮结果: Biden={votes['Biden']}, Trump={votes['Trump']}, Undecided={votes['Undecided']}")
        
        self.current_round += 1
    
    def run(self, rounds: int = 3):
        """运行模拟"""
        print(f"\n{'='*50}")
        print(f"开始选举模拟 - {self.num_voters} 位选民, {rounds} 轮辩论")
        print(f"{'='*50}")
        
        # 初始状态
        initial_votes = self.count_votes()
        print(f"\n初始状态: Biden={initial_votes['Biden']}, Trump={initial_votes['Trump']}, Undecided={initial_votes['Undecided']}")
        
        for _ in range(rounds):
            self.step()
        
        print(f"\n{'='*50}")
        print("模拟结束")
        print(f"{'='*50}")
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """获取完整结果"""
        return {
            'num_voters': self.num_voters,
            'rounds': self.current_round,
            'history': self.results_history,
            'final_votes': self.count_votes(),
            'agent_details': [
                {
                    'id': a.unique_id,
                    'profile': {
                        'category': a.profile.category.value,
                        'name': a.profile.name,
                        'age': a.profile.age,
                        'gender': a.profile.gender,
                        'race': a.profile.race,
                        'state': a.profile.state,
                        'location': a.profile.location_type,
                        'education': a.profile.education,
                        'income': a.profile.income_level,
                        'religion': a.profile.religion,
                        'background': a.profile.background_story[:200] + '...',  # 截断以节省空间
                        'issues': a.profile.key_issues,
                        'susceptibility': round(a.profile.susceptibility, 2)
                    },
                    'initial_vote': a.profile.initial_preference.value,
                    'vote_history': a.vote_history,
                    'final_vote': a.current_vote.value,
                    'reasoning_samples': a.reasoning_history[-2:] if a.reasoning_history else []
                }
                for a in self.schedule.agents
            ]
        }


def run_llm_election(num_voters: int = 10, rounds: int = 3):
    """
    运行 LLM 驱动的选举实验
    
    Args:
        num_voters: 选民数量
        rounds: 辩论轮数
    """
    print("初始化 LLM...")
    llm = create_default_llm()
    
    # 测试 LLM 连接
    test_response = llm.send_message("请回复'OK'")
    if not test_response:
        print("LLM 连接失败，请检查配置")
        return None
    print(f"LLM 连接成功: {test_response[:20]}...")
    
    # 创建并运行模型
    model = LLMElectionModel(num_voters=num_voters, llm=llm)
    results = model.run(rounds=rounds)
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'election', 'llm.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


def run_multi_model_comparison(num_voters: int = 10, rounds: int = 3, models: List[str] = None):
    """
    运行多模型对比实验
    
    Args:
        num_voters: 选民数量
        rounds: 辩论轮数
        models: 要对比的模型列表
    """
    if models is None:
        models = ['gpt-4o-mini', 'gpt-4o']
    
    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"测试模型: {model_name}")
        print(f"{'='*60}")
        
        try:
            # 使用新的 create_llm 函数（自动选择正确的 API 密钥）
            llm = create_llm(model=model_name)
            
            # 测试连接
            test = llm.send_message("回复OK")
            if not test:
                print(f"模型 {model_name} 连接失败，跳过")
                continue
            print(f"模型 {model_name} 连接成功")
            
            # 固定随机种子以确保可比性
            random.seed(42)
            
            # 运行模拟
            model = LLMElectionModel(num_voters=num_voters, llm=llm)
            results = model.run(rounds=rounds)
            
            all_results[model_name] = results
            
        except Exception as e:
            print(f"模型 {model_name} 出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出对比结果
    print(f"\n{'='*60}")
    print("多模型对比结果")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        votes = results['final_votes']
        print(f"\n{model_name}:")
        print(f"  Biden: {votes['Biden']} ({votes['Biden']/num_voters*100:.1f}%)")
        print(f"  Trump: {votes['Trump']} ({votes['Trump']/num_voters*100:.1f}%)")
        print(f"  Undecided: {votes['Undecided']} ({votes['Undecided']/num_voters*100:.1f}%)")
    
    # 保存对比结果
    output_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'election', 'multi_model.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n对比结果已保存到: {output_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM 选举实验（按 Pew 研究标准）')
    parser.add_argument('--voters', type=int, default=5, help='选民数量（建议5-20）')
    parser.add_argument('--rounds', type=int, default=3, help='辩论轮数')
    parser.add_argument('--compare', action='store_true', help='启用多模型对比')
    parser.add_argument('--models', nargs='+', default=['gpt-4o-mini'], 
                       help='要使用的模型（多模型对比时）')
    
    args = parser.parse_args()
    
    print(f"运行 LLM 选举实验: {args.voters} 位选民, {args.rounds} 轮辩论")
    print("选民配置: 基于 Pew Research 政治类型学")
    print("辩论内容: 2020 年大选核心议题（平衡双方观点）")
    print("注意: 每位选民每轮会调用一次 API\n")
    
    if args.compare:
        results = run_multi_model_comparison(
            num_voters=args.voters, 
            rounds=args.rounds,
            models=args.models
        )
    else:
        results = run_llm_election(num_voters=args.voters, rounds=args.rounds)
    
    if results and not args.compare:
        print("\n最终投票结果:")
        print(json.dumps(results['final_votes'], indent=2))

