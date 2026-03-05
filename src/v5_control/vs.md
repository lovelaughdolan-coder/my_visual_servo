我来帮你把这两篇视觉伺服控制的经典论文转换成简洁的Markdown格式，去掉冗长的实验数据和参考文献。

---

# Visual Servo Control Part I: Basic Approaches

**作者**: François Chaumette, Seth Hutchinson  
**来源**: IEEE Robotics & Automation Magazine, 2006

## 1. 引言

视觉伺服控制是指利用计算机视觉数据来控制机器人运动的技术。相机可以安装在机器人上（eye-in-hand），也可以固定在工作空间中（eye-to-hand）。本文主要讨论eye-in-hand配置。

## 2. 视觉伺服的基本组成

### 2.1 误差定义

所有基于视觉的控制方案目标是最小化误差：
$$e(t) = s(m(t), a) - s^*$$

其中：
- $m(t)$：图像测量（如特征点像素坐标）
- $s$：视觉特征向量（k维）
- $a$：系统参数（相机内参、3D模型等）
- $s^*$：期望特征值

### 2.2 核心关系式

特征变化率与相机速度的关系：
$$\dot{s} = L_s v_c$$

其中 $L_s \in \mathbb{R}^{k \times 6}$ 称为**交互矩阵**（Interaction Matrix）或特征雅可比矩阵。

误差动态：
$$\dot{e} = L_e v_c$$

### 2.3 控制律设计

采用速度控制，期望指数收敛 $\dot{e} = -\lambda e$：
$$v_c = -\lambda \widehat{L_e^+} e$$

其中 $\widehat{L_e^+}$ 是交互矩阵伪逆的估计值。

---

## 3. 经典图像视觉伺服（IBVS）

### 3.1 特征选择

使用图像平面点的坐标 $s = (x, y)$，其中：
$$x = \frac{X}{Z} = \frac{u - c_u}{f}, \quad y = \frac{Y}{Z} = \frac{v - c_v}{f}$$

### 3.2 交互矩阵推导

对于3D点 $(X, Y, Z)$ 投影到图像点 $(x, y)$，交互矩阵为：

$$L_x = \begin{bmatrix} 
-\frac{1}{Z} & 0 & \frac{x}{Z} & xy & -(1+x^2) & y \\
0 & -\frac{1}{Z} & \frac{y}{Z} & 1+y^2 & -xy & -x
\end{bmatrix}$$

**关键问题**：矩阵中含有深度 $Z$，必须估计或近似。

### 3.3 多特征点情况

控制6DOF至少需要3个点（$k \geq 6$）。将多个点的交互矩阵堆叠：
$$L_x = \begin{bmatrix} L_{x_1} \\ L_{x_2} \\ L_{x_3} \\ \vdots \end{bmatrix}$$

**注意**：3个点时 $L_x$ 可能奇异，且存在4个全局最小值无法区分，实际通常使用更多点。

### 3.4 交互矩阵的近似方法

| 方法 | 公式 | 特点 |
|:---|:---|:---|
| 当前深度 | $\widehat{L_e^+} = L_e^+$ | 需实时估计当前深度 $Z$ |
| 期望深度 | $\widehat{L_e^+} = L_{e^*}^+$ | 常数矩阵，仅需期望深度 $Z^*$ |
| 混合方法 | $\widehat{L_e^+} = \frac{1}{2}(L_e + L_{e^*})^+$ | 实践中表现最好 |

### 3.5 IBVS的几何解释

**纯旋转问题**：当绕光轴旋转接近 $\pi$ 时，IBVS会产生意外的后退平移运动。这是因为交互矩阵第3列和第6列的耦合。

- 小误差时：行为良好，图像轨迹接近直线
- 大误差时：可能产生非预期的笛卡尔轨迹，甚至收敛到局部最小值

---

## 4. 立体视觉IBVS

使用双目相机，特征为左右图像点坐标：
$$s = (x_l, y_l, x_r, y_r)$$

通过空间运动变换矩阵 $V$ 将左右相机速度转换到传感器坐标系：
$$L_{x_s} = \begin{bmatrix} L_{x_l} \cdot {}^lV_s \\ L_{x_r} \cdot {}^rV_s \end{bmatrix}$$

**注意**：由于极线约束，$L_{x_s} \in \mathbb{R}^{4 \times 6}$ 秩为3，至少需要3个点控制6DOF。

---

## 5. 基于位置的视觉伺服（PBVS）

### 5.1 特征定义

使用相机相对于参考坐标系的位姿：
$$s = ({}^c t_o, \theta u)$$

其中：
- ${}^c t_o$：物体坐标系原点在相机坐标系中的平移
- $\theta u$：旋转的角度-轴表示

### 5.2 两种PBVS方案

**方案1**：$s = ({}^c t_o, \theta u)$，误差 $e = ({}^c t_o - {}^{c^*} t_o, \theta u)$

交互矩阵：
$$L_e = \begin{bmatrix} R & 0 \\ 0 & L_{\theta u} \end{bmatrix}$$

控制律：
$$\begin{cases} v_c = -\lambda R^\top ({}^{c^*} t_o - {}^c t_o) \\ \omega_c = -\lambda \theta u \end{cases}$$

特点：相机轨迹为直线，但图像轨迹可能不理想。

**方案2**：$s = ({}^{c^*} t_c, \theta u)$，即相对于期望坐标系的平移

交互矩阵：
$$L_e = \begin{bmatrix} -I_3 & [{}^c t_o]_\times \\ 0 & L_{\theta u} \end{bmatrix}$$

控制律：
$$\begin{cases} v_c = -\lambda ({}^{c^*} t_o - {}^c t_o + [{}^c t_o]_\times \theta u) \\ \omega_c = -\lambda \theta u \end{cases}$$

特点：物体原点在图像中的轨迹为直线，但相机轨迹非直线。

---

## 6. 稳定性分析

### 6.1 李雅普诺夫分析

定义 $L = \frac{1}{2}\|e(t)\|^2$，其导数为：
$$\dot{L} = -\lambda e^\top L_e \widehat{L_e^+} e$$

**全局渐近稳定条件**：
$$L_e \widehat{L_e^+} > 0$$

### 6.2 IBVS稳定性

- $k > 6$ 时，$L_e \widehat{L_e^+} \in \mathbb{R}^{k \times k}$ 秩最多为6，存在非平凡零空间
- 只能保证**局部渐近稳定**
- 存在局部最小值（如图9所示，误差指数衰减但不为零）

局部稳定条件（变换误差 $e' = \widehat{L_e^+}e$）：
$$\widehat{L_e^+} L_e > 0$$

### 6.3 PBVS稳定性

- 当 $\theta \neq 2k\pi$ 时，$L_{\theta u}$ 非奇异
- 若位姿参数完美估计，$L_e L_e^{-1} = I_6$，**全局渐近稳定**
- 但实际中位姿估计受噪声和标定误差影响，可能失稳

---

## 7. IBVS vs PBVS 对比总结

| 特性 | IBVS | PBVS |
|:---|:---|:---|
| **传感器模型** | 2D传感器 | 3D传感器 |
| **特征空间** | 图像空间 | 笛卡尔空间 |
| **3D参数需求** | 仅需深度Z | 需要完整位姿 |
| **对噪声鲁棒性** | 强 | 弱 |
| **相机轨迹** | 可能非最优 | 理论最优（直线） |
| **图像轨迹** | 直线 | 可能出视野 |
| **稳定性** | 局部渐近稳定 | 全局渐近稳定（理想情况） |
| **主要问题** | 局部最小值、奇异点 | 位姿估计误差敏感 |

**核心结论**：没有绝对优越的方法，只有性能权衡。IBVS对校准误差和图像噪声鲁棒，但可能产生非预期笛卡尔轨迹；PBVS在笛卡尔空间轨迹最优，但依赖精确的3D位姿估计。

---

# Visual Servo Control Part II: Advanced Approaches

**作者**: François Chaumette, Seth Hutchinson  
**来源**: IEEE Robotics & Automation Magazine, 2007

## 1. 3D参数估计方法

### 1.1 极线几何与单应性矩阵

给定当前图像和期望图像的匹配点，可估计：
- **基本矩阵**（未标定相机）或**本质矩阵**（已标定相机）
- **单应性矩阵** $H$：$x_i = H x_i^*$

单应性分解：
$$H = R + \frac{t}{d^*} n^{*\top}$$

可恢复 $R$、$t/d^*$、$n^*$（两个解，需根据期望姿态选择）。

**问题**：接近收敛时（两图像相似），极线几何退化，估计不准确。

### 1.2 交互矩阵的直接估计

通过观测特征变化与相机运动的关系：
$$L_s \Delta v_c = \Delta s$$

使用 $N > 6$ 次独立运动，最小二乘解：
$$\widehat{L_s} = B A^+$$

或直接估计伪逆：
$$\widehat{L_s^+} = A B^+$$

**在线估计**：Broyden更新规则
$$\widehat{L_s}(t+1) = \widehat{L_s}(t) + \frac{\alpha}{\Delta v_c^\top \Delta v_c}(\Delta s - \widehat{L_s}(t)\Delta v_c)\Delta v_c^\top$$

**优缺点**：避免建模和标定，但缺乏理论稳定性分析。

---

## 2. 混合视觉伺服（Hybrid VS）

结合IBVS和PBVS的优势，实现平移与旋转解耦。

### 2.1 2.5D视觉伺服

特征选择：
$$s_t = (x, \log Z), \quad s = (s_t, \theta u)$$

其中 $\rho_Z = Z/Z^*$ 可从部分位姿估计获得。

交互矩阵：
$$L_v = \frac{1}{Z^* \rho_Z} \begin{bmatrix} -1 & 0 & x \\ 0 & -1 & y \\ 0 & 0 & -1 \end{bmatrix}, \quad L_\omega = \begin{bmatrix} xy & -(1+x^2) & y \\ 1+y^2 & -xy & -x \\ -y & x & 0 \end{bmatrix}$$

控制律：
$$v_c = -L_v^+ (\lambda e_t + L_\omega \omega_c)$$

**优势**：图像轨迹为直线，相机速度平滑，全局渐近稳定。

### 2.2 其他混合方案

- **方案2**：$s = ({}^{c^*} t_c, x_g, \theta u_z)$，同时实现3D直线轨迹和2D直线轨迹
- **方案3**：使用图像点坐标乘以深度 $s = (u_1 Z_1, v_1 Z_1, Z_1, ...)$，冗余特征避免局部最小值

---

## 3. 分区视觉伺服（Partitioned VS）

目标：仅使用图像特征实现各自由度解耦。

### 3.1 光轴运动分离

将交互矩阵分区：
$$\dot{s} = L_{xy} v_{xy} + L_z v_z$$

其中 $v_z = (v_z, \omega_z)$ 为沿光轴的平移和旋转。

定义新特征：
- $\alpha$：两点连线与水平轴的夹角（关联 $\omega_z$）
- $\sigma^2$：多边形面积（关联 $v_z$）

控制：
$$\begin{cases} v_z = \lambda_{v_z} \ln \frac{\sigma^*}{\sigma} \\ \omega_z = \lambda_{\omega_z} (\alpha^* - \alpha) \end{cases}$$

### 3.2 柱坐标IBVS

使用图像点的柱坐标 $(\rho, \theta)$：
$$\rho = \sqrt{x^2 + y^2}, \quad \theta = \arctan \frac{y}{x}$$

交互矩阵：
$$L_\rho = \begin{bmatrix} -\frac{c}{Z} & -\frac{s}{Z} & \frac{\rho}{Z} & \frac{\rho}{Z} & (1+\rho^2)s & -(1+\rho^2)c & 0 \end{bmatrix}$$
$$L_\theta = \begin{bmatrix} \frac{s}{\rho Z} & -\frac{c}{\rho Z} & 0 & c/\rho & s/\rho & -1 \end{bmatrix}$$

**优势**：$\rho$ 与 $\omega_z$ 解耦，$\theta$ 与 $v_z$ 解耦且与 $\omega_z$ 线性相关。

---

## 4. 高级IBVS方案

### 4.1 其他几何基元

除点特征外，可使用：
- 线段、直线
- 圆、圆柱
- **图像矩**（Image Moments）：适用于任意形状平面物体

通过选择合适的矩组合，可实现良好的解耦和线性化特性。

### 4.2 性能优化与规划

**最优控制**：LQG设计，平衡跟踪误差与机器人运动。

**可分辨性（Resolvability）**：利用交互矩阵的奇异值分解，分析各自由度的可见性。

**梯度投影法**：将次要约束（如关节限位）投影到视觉任务的零空间：
$$e_g = \widehat{L_e^+} e + P_e e_s$$

其中 $P_e = (I_6 - \widehat{L_e^+} \widehat{L_e})$。

---

## 5. 切换控制方案

结合IBVS和PBVS的切换策略：

1. 初始使用IBVS，监控PBVS的李雅普诺夫函数 $\mathcal{L}_P = \frac{1}{2}\|e_P\|^2$
2. 若 $\mathcal{L}_P > \gamma_P$，切换到PBVS
3. 在PBVS模式下，若IBVS的李雅普诺夫函数 $\mathcal{L}_I > \gamma_I$，切换回IBVS

**效果**：大旋转时先IBVS（特征直线运动），接近时PBVS（避免相机后退），实现优势互补。

---

## 6. 目标跟踪

对于运动目标，误差动态：
$$\dot{e} = L_e v_c + \frac{\partial e}{\partial t}$$

控制律需补偿目标运动：
$$v_c = -\lambda \widehat{L_e^+} e - \widehat{L_e^+} \widehat{\frac{\partial e}{\partial t}}$$

**目标运动估计方法**：
- 积分项：$\widehat{\frac{\partial e}{\partial t}} = \mu \sum_j e(j)$（仅对恒定速度有效）
- 前馈估计：$\widehat{\frac{\partial e}{\partial t}} = \widehat{\dot{e}} - \widehat{L_e} \widehat{v_c}$
- 卡尔曼滤波或更复杂的滤波方法
- 利用目标轨迹的先验知识（如周期性运动预测）

---

## 7. 关节空间控制

当机器人自由度 $n \neq 6$ 或需考虑关节限位时，在关节空间设计控制：

$$\dot{s} = J_s \dot{q} + \frac{\partial s}{\partial t}$$

其中特征雅可比矩阵 $J_s \in \mathbb{R}^{k \times n}$：

**Eye-in-hand**：
$$J_s = L_s {}^c V_N {}^N J(q)$$

**Eye-to-hand**：
$$J_s = -L_s' {}^c V_0 {}^0 J(q)$$

控制律：
$$\dot{q} = -\lambda \widehat{J_e^+} e - \widehat{J_e^+} \widehat{\frac{\partial e}{\partial t}}$$

**稳定性条件**：
- $k = n$：$J_e \widehat{J_e^+} > 0$（全局渐近稳定）
- $k > n$：$\widehat{J_e^+} J_e > 0$（局部渐近稳定）

---

## 8. 结论与展望

本文讨论的先进方法旨在补偿基本IBVS和PBVS的相对缺陷：

| 方法类别 | 核心思想 | 主要优势 |
|:---|:---|:---|
| 3D参数估计 | 极线几何/直接估计 | 减少标定依赖 |
| 混合视觉伺服 | 2D+3D特征组合 | 解耦平移旋转，全局稳定 |
| 分区视觉伺服 | 图像特征特殊设计 | 纯2D特征实现解耦 |
| 切换控制 | IBVS/PBVS动态选择 | 结合两者优势 |
| 目标跟踪 | 运动补偿 | 处理动态场景 |
| 关节空间控制 | 考虑机器人运动学 | 处理冗余/欠驱动系统 |

**未来方向**：高速任务需考虑机器人动力学、非完整约束机器人、其他视觉传感器（鱼眼、全景、超声）、多传感器融合（力觉、接近觉）等。

---

这两篇是视觉伺服领域的经典教程，Part I建立基础概念，Part II深入高级方法。需要我针对某个具体部分（比如2.5D视觉伺服的公式推导，或者柱坐标IBVS的实现细节）再展开说明吗？