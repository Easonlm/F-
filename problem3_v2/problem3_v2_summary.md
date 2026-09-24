# 问题三 V2 摘要

V1 保留不变。V2 精化结构点、核查 ALR 配比支持、联合重采样八个 Model C 参数，并给出 E0/E1/E2 与 36 场景 minimax regret。

P1 观测配比在 λ=1 下预测改善 0.5022；λ 随 N 衰减时收益按情景降低，尚无跨尺度联合识别。Loose 配比的支持等级：extrapolative。

结构点：exponential/start: 3.7024e+18; exponential/saturation: 1.377e+20; power/start: 1.2912e+19; power/saturation: 7.215e+19; logarithmic/start: 2.0036e+18; logarithmic/saturation: 2.0036e+18。

高预算 10^24 的三级结果：E0_empirical: N=11.966, D=299.89, Q=1, Loss=2.0934, used=2.6%; E1_moderate_3x: N=35.897, D=899.68, Q=1, Loss=1.9794, used=22.4%; E2_free_scaling: N=39.562, D=3657, Q=1, Loss=1.916, used=100.0%。

联合 bootstrap 的 Q=1 稳定率：1e+19:0.0%; 1e+20:10.5%; 1e+22:100.0%; 1e+24:100.0%。理论弹性 eN=0.4515、eD=0.5485。稳健推荐使用 E1 3× 敏感性边界内的 minimax regret 解；3× 不是实证支持。
