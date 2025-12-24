# 贡献指南

感谢您对 Casevo 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议，请：

1. 检查 [Issues](https://github.com/rgCASS/casevo/issues) 是否已有相关问题
2. 创建新 Issue，提供：
   - 问题描述
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（Python 版本、操作系统等）

### 提交代码

1. **Fork 仓库**
2. **创建分支**: `git checkout -b feature/your-feature-name`
3. **编写代码**: 遵循代码规范
4. **添加测试**: 为新功能添加测试
5. **提交更改**: `git commit -m "Add: your feature description"`
6. **推送分支**: `git push origin feature/your-feature-name`
7. **创建 Pull Request**

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型提示
- 添加文档字符串
- 编写单元测试
- 保持代码简洁可读

### 文档贡献

- 更新相关文档
- 添加代码示例
- 修正错误和拼写
- 改进文档结构

## 开发环境设置

1. Fork 并克隆仓库
2. 创建虚拟环境
3. 安装开发依赖：
   ```bash
   pip install -e ".[experiments]"
   pip install pytest pytest-cov black flake8 mypy
   ```
4. 运行测试：
   ```bash
   pytest tests/
   ```

## 提交信息规范

使用清晰的提交信息：

- `Add: 新功能描述`
- `Fix: 修复的问题描述`
- `Update: 更新的内容描述`
- `Docs: 文档更新描述`
- `Refactor: 重构内容描述`

## 许可证

贡献的代码将采用与项目相同的 MIT 许可证。

## 联系方式

如有问题，请通过 GitHub Issues 联系。

