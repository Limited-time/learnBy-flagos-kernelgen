# learnBy-flagos-kernelgen
KernelGen：An AI kernel generator and compiler for cross-architecture deployment。

本项目记录关于探索KernelGen工程中的形成的案例与不完整思考

（1）KernelGen_test

使用kernelGen生成算子

- broadcast：广播操作本质是将低维张量的每个元素复制到高维的对应位置，每个输出元素仅关联低维张量中一个元素，符合逐点特征。

（2）flagGems_chat

- 对flagGems工程的解读chat

