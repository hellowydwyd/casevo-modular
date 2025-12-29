"""
实验配置加载器

支持从 JSON 配置文件加载实验参数。
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigLoader:
    """
    配置加载器
    
    负责从 JSON 文件加载实验配置。
    """
    
    # 默认配置目录
    DEFAULT_CONFIG_DIR = os.path.join(
        os.path.dirname(__file__), '..', 'configs'
    )
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置目录路径，默认为 experiments/configs
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load(self, config_name: str, 
             variant: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_name: 配置名称（不含 .json 后缀）
            variant: 变体名称（如 'baseline', 'optimized'）
            
        Returns:
            配置字典
        """
        # 检查缓存
        cache_key = f"{config_name}:{variant or 'all'}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # 构建文件路径
        if not config_name.endswith('.json'):
            config_name = f"{config_name}_config.json"
        
        filepath = os.path.join(self.config_dir, config_name)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        
        # 加载配置
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 如果指定了变体，返回该变体的配置
        if variant:
            if variant not in config:
                raise KeyError(f"配置变体不存在: {variant}")
            result = config[variant]
        else:
            result = config
        
        # 缓存结果
        self._cache[cache_key] = result
        
        return result.copy()
    
    def load_election_config(self, variant: str = 'baseline') -> Dict[str, Any]:
        """
        加载选举实验配置
        
        Args:
            variant: 变体名称
            
        Returns:
            配置字典
        """
        return self.load('election', variant=variant)
    
    def load_resource_config(self, variant: str = 'baseline') -> Dict[str, Any]:
        """
        加载资源分配实验配置
        
        Args:
            variant: 变体名称
            
        Returns:
            配置字典
        """
        return self.load('resource', variant=variant)
    
    def load_info_spreading_config(self, variant: str = 'baseline') -> Dict[str, Any]:
        """
        加载信息传播实验配置
        
        Args:
            variant: 变体名称
            
        Returns:
            配置字典
        """
        return self.load('info_spreading', variant=variant)
    
    def list_configs(self) -> List[str]:
        """
        列出所有可用配置
        
        Returns:
            配置文件名列表
        """
        configs = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                configs.append(filename)
        return configs
    
    def list_variants(self, config_name: str) -> List[str]:
        """
        列出配置的所有变体
        
        Args:
            config_name: 配置名称
            
        Returns:
            变体名称列表
        """
        config = self.load(config_name)
        
        # 排除非变体字段
        exclude_keys = {
            'experiment_name', 'description', 'evaluation_metrics', 
            'political_distribution', 'run_settings',
            # info_spreading 配置的元数据字段
            'agent_type_distribution', 'network_settings', 'agent_attributes',
            # resource 配置的元数据字段
            'priority_distribution', 'need_ranges'
        }
        
        return [k for k in config.keys() if k not in exclude_keys]
    
    def clear_cache(self):
        """清空配置缓存"""
        self._cache.clear()


# 全局配置加载器实例
_default_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """
    获取默认配置加载器
    
    Returns:
        ConfigLoader 实例
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader


def load_config(config_name: str, variant: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：加载配置
    
    Args:
        config_name: 配置名称
        variant: 变体名称
        
    Returns:
        配置字典
    """
    return get_config_loader().load(config_name, variant)


def load_election_config(variant: str = 'baseline') -> Dict[str, Any]:
    """便捷函数：加载选举实验配置"""
    return get_config_loader().load_election_config(variant)


def load_resource_config(variant: str = 'baseline') -> Dict[str, Any]:
    """便捷函数：加载资源分配实验配置"""
    return get_config_loader().load_resource_config(variant)


def load_info_spreading_config(variant: str = 'baseline') -> Dict[str, Any]:
    """便捷函数：加载信息传播实验配置"""
    return get_config_loader().load_info_spreading_config(variant)

