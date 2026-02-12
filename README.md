# learnBy-flagos-kernelgen
KernelGen：An AI kernel generator and compiler for cross-architecture deployment。

本项目记录关于实践KernelGen工具中的形成的案例与不完整思考

（1）KernelGen_test

使用kernelGen生成算子

- broadcast：广播操作本质是将低维张量的每个元素复制到高维的对应位置，每个输出元素仅关联低维张量中一个元素，符合逐点特征。

（2）flagGems_chat

- 对flagGems工程的解读chat



## 说明

**broadcast**算子：参考并引用”昇腾AI算法挑战赛高阶赛-Broadcast算子优化任务书.pdf“

flagGems_chat中的README_cn_manual.md基于FlagGems/README_cn.md修改

本文档及项目遵循 **Apache License 2.0** 开源协议。

**Supported by Upstream Labs｜源起之道支持**

