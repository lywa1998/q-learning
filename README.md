# q-learning

随机过程作业

# 文件
```angular2html
.
├── agents
│   ├── __init__.py
│   ├── Q_Leaning.py # Q-Learning智能体
│   └── Sarsa.py # Sarsa智能体
├── LICENSE
├── main.py # 主函数
├── README.md
├── save
│   ├── model-qlearning.npy # Q-Learning训练结果
│   └── model-sarsa.npy # Sarsa训练结果
└── visual.py
```

# 运行
```bash
# 训练
python --agent [qlearning | sarsa] --train

# 验证
python --agent [qlearning | sarsa]
```

# TODO
- [X] Train代码
- [x] Test代码