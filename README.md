# Kirigami

Kirigami 是用于机器人演示数据的桌面标注工具，支持读取 Prometheus 原始 episode（`prometheus_raw_episode_v1`），同步查看三路相机与双臂 3D 姿态，手动修剪首尾静止帧，再按子任务导出短 episode。

推荐工作顺序：**安装环境 → 打开 GUI → 导入 episode → Trim 首尾 → 标注子任务分段 → 导出 Slice**。

本文使用一个完整例子：原始 episode 有 **1,200 帧**，去掉首尾各 **100 帧**，再将剩下的 **1,000 帧**分成 `approach`、`pick`、`place`、`return` 四个子任务。所有 `/data/robot_demo/...` 都是示例数据路径，请换成自己的路径。

| 操作 | GUI 按钮 | 保存结果 |
| --- | --- | --- |
| Trim：手动去掉首尾静止部分 | `Keep in`、`Keep out`、`Trim source…` | 确认后写回原 episode 目录，并保留完整原始备份 |
| Slice：按子任务分段导出 | `+ Split M`、`Export segments →` | 在导出目录生成独立的短 episode，保留源 episode 的视频和观测 |

阅读导航：[安装环境](#安装环境) · [激活环境](#激活环境) · [打开 GUI](#打开-gui) · [Trim 示例](#示例一-trim-首尾静止帧) · [添加子任务名](#添加新的子任务名) · [Slice 示例](#示例二-slice-成四个子任务) · [常见问题](#常见问题)

## 安装环境

### 准备条件

- 主机已安装 Miniconda 或 Anaconda。本机的 Conda 安装目录是 `$HOME/miniconda3`。
- 使用已登录的 Linux 图形桌面，在该桌面的终端中启动 GUI。支持 X11 / Wayland 会话；3D 视图需要桌面驱动提供 OpenGL 3.3 或以上。
- 首次安装需要能够下载 Conda / pip 依赖。Trim 和导出时需要额外磁盘空间保存裁剪结果。

进入仓库，执行安装脚本：

```bash
cd /home/romoya-2/workspace/kirigami
./setup_env.sh
```

如果仓库放在其他位置，请修改 `cd` 后面的路径。

脚本根据 [environment.yml](environment.yml) 创建独立的 `kirigami` 环境；环境已存在时会更新它。主要依赖包括 Python 3.11、PySide6、FFmpeg、NumPy、SciPy、trimesh，以及 Qt 在 Linux 上需要的 XCB 库。

环境安装以 `setup_env.sh` / `environment.yml` 为准。[requirements.txt](requirements.txt) 只描述 pip 依赖层，不能替代完整的 Conda 环境安装。

依赖中的 `opencv-python-headless` 用于视频处理，桌面窗口由 PySide6 提供。Kirigami 本身仍然需要真实桌面显示。

## 激活环境

安装脚本结束后，当前终端不会自动切换环境。需要手动运行 Python 或测试时，在终端执行：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate kirigami
python --version
python -c "import sys; print(sys.executable)"
```

应看到 Python 3.11，以及类似下面的解释器路径：

```text
/home/romoya-2/miniconda3/envs/kirigami/bin/python
```

如果 Conda 安装在其他位置，例如 `$HOME/anaconda3`，请调整 `source` 的路径。已经初始化过 Conda 的终端可以直接执行 `conda activate kirigami`。每个新终端需要分别激活；用完后可运行 `conda deactivate`。

日常启动也可以直接使用下面的 `run_kirigami.sh`：它会在自己的进程中激活 `kirigami` 环境，并设置 Qt 所需的库路径。

## 打开 GUI

### 先检查桌面和显卡

在主机图形桌面的终端中执行：

```bash
cd /home/romoya-2/workspace/kirigami
./run_kirigami.sh --check
```

`--check` 会打开一个可见窗口，检查 3D 模型和实际 OpenGL 渲染，打印结果后自动退出。成功时会包含：

```text
visible=True, robot3d=True, opengl=True
3D renderer: NVIDIA GeForce RTX 5090/PCIe/SSE2; triangles=299,312
```

上面的显卡名称来自本机验证结果；其他主机显示其实际渲染设备。

### 启动工作界面

```bash
./run_kirigami.sh
```

窗口打开后，点击右上角 **Open folder…** 或左侧 **+ Add episodes** 导入数据。可以选择整个任务目录，也可以只选择一个 episode。

例如，数据存放为：

```text
/data/robot_demo/demo_task/
├── 0007/
│   ├── camera/
│   │   ├── base_0_rgb.mp4
│   │   ├── left_wrist_0_rgb.mp4
│   │   ├── right_wrist_0_rgb.mp4
│   │   ├── timestamp.pkl
│   │   └── metadata.json
│   ├── robot/robot_state_dict.pkl
│   ├── action/executed_action_dict.pkl
│   ├── event/event_dict.pkl
│   └── manifests/
└── 0008/
    └── ...
```

也可以启动时直接指定目录：

```bash
# 导入整个任务目录
./run_kirigami.sh /data/robot_demo/demo_task
```

```bash
# 只导入 episode 0007
./run_kirigami.sh /data/robot_demo/demo_task/0007
```

`event/` 和部分元数据可以缺省。正常切片导出需要三路相机视频、相机时间戳、机器人状态和动作数据；具体支持范围见[数据范围与同步方式](#数据范围与同步方式)。

如果使用已经激活的环境直接运行 Python 入口，先配置运行库路径：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m kirigami.app
```

需要应用菜单入口时，在仓库目录运行一次：

```bash
./install_desktop.sh
```

之后在主机应用菜单中搜索 **Kirigami**。代码更新后请重新打开程序；执行 Trim 前，关闭仍在使用同一 episode 的旧版窗口。

### 认识工作区

| 区域 | 用途 |
| --- | --- |
| 左侧 `Episodes` | 选择 episode，查看帧数和导出状态 |
| 中间四个视图 | `Overview`、`Robot 3D`、`Left wrist`、`Right wrist` 同步查看 |
| `Timeline` 与 `FRAME` | 播放、逐帧定位、添加分界点、选择 Trim 范围 |
| `Joint signals` | 辅助观察机器人关节变化；点击曲线可跳转 |
| 右侧 `Segments` | 给每段选择 `Subtask`，勾选需要导出的 `Export` |
| 右侧 `SUBTASK PRESETS` | 添加可复用的子任务名称 |
| 左侧 `EXPORT DESTINATION` | 查看或选择切片输出目录 |

## 示例一 Trim 首尾静止帧

假设 `demo_task/0007` 有 **1,200 帧**，帧号从 **0 到 1199**。观看后，认为 `0–99` 和 `1100–1199` 是不需要的首尾静止部分，希望保留 **100–1099**。

1. 导入并选中 `0007`，播放视频，结合关节曲线确认首尾范围。静止与否由你手动判断。
2. 在 `FRAME` 输入 **100**，按回车定位，然后点击 **Keep in I**。
3. 在 `FRAME` 输入 **1099**，按回车定位，然后点击 **Keep out O**。
4. 检查时间线上被遮罩的待删除部分，以及下面的范围说明：

   ```text
   Keep 100–1099 · 1,000 frames
   Remove head 100 / tail 100
   ```

5. 点击 **Trim source…**，核对弹窗中的原始目录、保留帧数、删除帧数和备份位置，确认后等待完成。

**Keep in 和 Keep out 选中的两帧都会保留。** 使用 `I` / `O` 快捷键时，请先离开名称或帧号输入框；直接点击按钮也可以。

完成后：

- 裁剪后的数据仍位于 `/data/robot_demo/demo_task/0007/`，GUI 自动重新加载该 episode。
- 新 episode 有 **1,000 帧**，显示帧号变为 **0–999**：新第 0 帧对应裁剪前第 100 帧，新第 999 帧对应裁剪前第 1099 帧。
- 相机视频、观测和对应时间戳同步裁剪，相关样本计数同步更新。数据中的绝对时间戳保持原值。
- 原始 1,200 帧的完整 episode 保存在独立备份目录：

  ```text
  /data/robot_demo/demo_task/.kirigami-backups/0007/<时间与唯一标识>/original/
  ```

只设置 `Keep in` / `Keep out` 会保存待执行的边界标签；**点击 Trim source 并确认后才改写原数据**。`Reset trim` 用于清除尚未执行的边界选择，不能撤销已经完成的裁剪。恢复已裁剪数据的方法见[备份与恢复](#备份与恢复)。

如果之前已经标好了子任务分段，Trim 会保留与新范围相交的标签、调整其帧号，并移除范围外的分段；已导出的切片需要重新导出。因此建议先完成 Trim，再做下面的子任务标注。

## 添加新的子任务名

继续使用 Trim 后的 1,000 帧。假设要标注四个子任务：

| 名称 | 示例含义 |
| --- | --- |
| `approach` | 接近物体 |
| `pick` | 抓取物体 |
| `place` | 放置物体 |
| `return` | 返回准备位置 |

### 方法一：在 GUI 中添加预设

1. 找到右侧 **SUBTASK PRESETS**。
2. 在 **New subtask name** 输入 `approach`，点击 **Add**。
3. 依次添加 `pick`、`place`、`return`。
4. 分段后，在 `Segments` 每一行的 **Subtask** 下拉框中选择对应名称。

**Add 只添加可复用的名称，不会自动给某一段赋值。** 预设默认保存到仓库中的 [subtasks.yaml](subtasks.yaml)，下次启动仍可使用。

### 随时重命名或删除预设

点击 **SUBTASK PRESETS** 标题旁的 **Manage…**，即使还没导入 episode，也可以管理词条：

| 操作 | 示例步骤 |
| --- | --- |
| 新增 | 在输入框填写 `check_grasp`，点击 **Add new**。 |
| 重命名 | 选中 `pick`，将输入框改为 `grasp`，点击 **Rename**。双击词条或按 **F2** 可快速进入名称编辑。 |
| 删除 | 选中 `reset`，点击 **Delete selected**，也可以在列表中按 **Delete**。 |
| 批量删除 | 按住 **Ctrl** 或 **Shift** 选择多个词条，再点击 **Delete selected**。可以删除全部预设，之后重新添加。 |

每次操作都会立即保存到预设文件，并同步更新分段行的下拉选项；点击 **Close** 关闭管理窗口即可。重名或不合法的名称会显示错误提示。

**这些操作只修改预设词条库，已有分段标签和导出的文件保持原样。** 例如将预设 `pick` 重命名为 `grasp` 后，已标为 `pick` 的分段仍叫 `pick`；需要修改时，在该分段的 **Subtask** 中选择 `grasp`，再重新导出。删除某个预设后，使用它的旧分段仍显示原标签，但其他行的预设选项中不再提供这个名称。只聚焦旧标签、切换行或重新打开 episode，都不会把已删除的名称自动加回预设。

如果提示预设文件已被其他窗口修改，关闭并重新打开 **Manage…** 读取最新词条后，再继续编辑。

### 方法二：直接在分段行中输入

`Segments` 表格中的 `Subtask` 下拉框也可以直接编辑。例如在某一行输入 `check_grasp`，然后按 **Tab** 或点击其他控件完成编辑。

当前分段的标签会自动保存；完成编辑后，新名称也会加入预设文件。程序不会把输入过程中的 `c`、`ch` 等半成品名称逐个加入预设。

### 方法三：编辑 YAML，或为不同项目指定预设

也可以关闭 GUI，在现有 `subtasks:` 列表下追加名称。例如保留仓库原有预设，再增加本例的四个名称：

```yaml
subtasks:
  - pick_up_egg
  - drag_egg
  - place_egg
  - reset
  - approach
  - pick
  - place
  - return
```

保存后重新打开 GUI。为不同项目使用独立预设文件时，可以指定 `--subtasks`：

```bash
./run_kirigami.sh /data/robot_demo/demo_task \
  --subtasks /data/robot_demo/demo_subtasks.yaml
```

此时 GUI 的 **Add**、**Manage…** 中的新增、重命名、删除，以及分段行中新名称的提交，都会保存到指定文件。

子任务名会用作输出文件夹名称，必须非空，不能包含 `/`、`\` 或控制字符，也不能是 `.`、`..`。例如 `pick_object` 合法，`pick/object` 不合法。同名子任务会归到同一任务目录，建议统一命名习惯。

## 示例二 Slice 成四个子任务

本例接着使用 **Trim 后的 1,000 帧**。接下来的帧号都以 Trim 后的 episode 为准。

### 添加三个分界点

1. 在 `FRAME` 输入 **200**，按回车定位，点击 **+ Split M**。
2. 同样在 **500**、**800** 帧添加分界点。
3. 右侧 `Segments` 会出现四行，分别选择四个子任务名，并勾选每行的 **Export**。

结果应为：

| 分段序号 | In | Out | 实际包含的帧 | 帧数 | Subtask | Export |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 200 | 0–199 | 200 | `approach` | 勾选 |
| 1 | 200 | 500 | 200–499 | 300 | `pick` | 勾选 |
| 2 | 500 | 800 | 500–799 | 300 | `place` | 勾选 |
| 3 | 800 | 1000 | 800–999 | 200 | `return` | 勾选 |

**切片采用 `[In, Out)`：包含 In，不包含 Out。** 因而第 200 帧属于 `pick` 段，边界帧不会重复。最后一段的 `Out=1000` 表示到 episode 末尾，最后实际保留的帧是 999。注意，这与 Trim 的 `Keep out` 含末帧的界面约定不同。

可以按 `M` 在当前帧添加分界点；重复按 `M` 不会删除已有点。删错或多加分界点时，将播放位置移到附近，点击 **Remove** 或按 **Delete**，程序会移除距离当前位置最近的分界点。

取消某行的 **Export** 只会跳过该段的导出，源 episode 中的这段数据继续保留。所有勾选 Export 的分段都必须填好 Subtask。

### 选择目录并导出

未指定输出目录时，本例默认写入：

```text
/data/robot_demo/demo_task_sliced/
```

需要改位置时，点击左侧 **EXPORT DESTINATION → Choose folder…**。也可以在启动时指定：

```bash
./run_kirigami.sh /data/robot_demo/demo_task \
  --out-root /data/robot_demo/annotated
```

确认四行名称和 Export 选项后，点击 **Export segments →**。程序会在后台导出并显示进度，全部切片通过校验后再发布到输出目录。导出结束后，当前 episode 会标为已导出，并尝试切换到后面的待处理 episode。

### 四个子任务会保存成什么样

使用默认输出目录时：

```text
/data/robot_demo/demo_task_sliced/
├── approach/
│   └── 0007_<来源标识>_00/
├── pick/
│   └── 0007_<来源标识>_01/
├── place/
│   └── 0007_<来源标识>_02/
└── return/
    └── 0007_<来源标识>_03/
```

每段是一个独立的短 episode。例如 `pick` 对应的文件夹中包含：

```text
0007_<来源标识>_01/
├── camera/
│   ├── base_0_rgb.mp4
│   ├── left_wrist_0_rgb.mp4
│   ├── right_wrist_0_rgb.mp4
│   ├── timestamp.pkl
│   └── metadata.json
├── robot/
│   └── robot_state_dict.pkl
├── action/
│   └── executed_action_dict.pkl
├── event/
│   └── event_dict.pkl
├── manifests/
└── slice_meta.json
```

本例中，`pick` 的三路相机各有 **300 帧**；机器人状态和动作按对应时间范围筛选，条数由实际采样频率决定。`slice_meta.json` 记录子任务名称、源 episode 路径、`start_frame=200`、`end_frame=500`、`n_frames=300` 和起止时间戳。这些帧号相对于 Trim 后的源 episode；回溯 Trim 前的范围可以查看源 episode 的 `annotations/trim_history.json`。

输出名称中的 `<来源标识>` 是源 episode 绝对路径生成的 12 位哈希，用于区分不同任务目录中同名的 `0007`。`00`–`03` 是原分段表中的序号；跳过一段时，后续序号不会重新排列。同一个子任务名可以对应多个独立的 episode 文件夹。

标注自动保存在源 episode 的 `annotations/slices.json`。修改分界点、子任务名或 Export 选项后，已导出状态会变为待处理；再次导出到同一输出根目录，会替换该源 episode 的旧切片，其他源 episode 的切片继续保留。

## 数据范围与同步方式

| 数据 | Trim 写回原目录 | Slice 导出短 episode |
| --- | --- | --- |
| 三路主相机视频与时间戳 | 支持 | 支持 |
| 标准 `robot/`、`action/`、`event/` 数据 | 支持 | 支持标准的时间戳对齐列表字段 |
| 相机元数据、manifest 样本计数 | 更新，保留原任务描述 | 更新，并写入对应子任务描述 |
| 独立 `observation/` 或 `observations/` 数据 | 支持 PKL、JSON、NPY、NPZ 的规定结构 | **目前不会导出这些附加文件** |
| 附加相机视频 | 匹配相机键后裁剪，保留原相对路径 | 裁剪已识别的附加视频，平铺保存到 `camera/` |

**如果训练依赖独立 observation 文件或自定义嵌套观测数组，需要先适配 Slice 导出逻辑。** 当前短 episode 的标准机器人观测来自 `robot/robot_state_dict.pkl`；Trim 对附加格式的支持不代表 Slice 已覆盖同样的格式。

相机按分界点对应的帧区间截取，机器人状态、动作和事件按该区间的 Unix 毫秒时间戳筛选，不要求它们与相机具有相同的采样频率。保留的数据时间戳不会归零；重新编码的视频自身播放时间从 0 开始。

切片导出要求三路主相机时间戳数量一致、相机时间戳严格递增，并且每段包含机器人和动作样本。视频逐帧裁剪后会重新编码，可能引入压缩损失；程序会通过解码核对输出帧数。

Trim 中，独立观测表带有 `timestamps` 时按时间筛选；没有时间戳的数组或表必须与相机一帧一行对应。无法识别的文件、采样数量不匹配或源目录中的符号链接会阻止写回，并显示相关路径。

## 备份与恢复

每次 Trim 都会生成独立备份，目录不会自动加入 episode 队列。`annotations/trim_history.json` 记录裁剪范围和备份位置。

恢复一次已完成的 Trim 时：

1. 关闭正在使用该 episode 的窗口。
2. 将当前 episode 整个目录移到其他位置，保留现有结果。
3. 将对应备份的 `original/` **完整复制**回原 episode 路径，再重新导入。

恢复时应一起恢复视频、观测、时间戳和标注，避免混用两个版本的数据。当前 GUI 没有一键恢复按钮。

<details>
<summary>写回方式、中断恢复与多窗口处理</summary>

Trim 先准备和校验完整的新 episode，再通过 Linux `renameat2(RENAME_EXCHANGE)` 交换原目录与准备好的目录。文件系统不支持该操作，或交换前处理失败时，源数据保持原样。交换后发生目录同步或日志错误时，程序会报告“裁剪已应用”，并保留原始备份。

每个备份的上一级目录中有 `transaction.json`。进程被强制终止或断电后，不要直接清理未完成的事务目录：`prepared` 状态可能位于目录交换的前后两侧。源 episode 的 Trim 历史中存在同一事务 ID，说明该次裁剪已应用，此时 `original/` 保存旧数据；否则应先检查两份目录，再确定恢复方式。

当前版本通过主机上的 episode 锁协调读取、导出和 Trim。检测到源数据已被另一个窗口裁剪时，会重新加载，并将旧窗口尚未保存的标签保存在 `.kirigami-backups/stale-labels/`，避免用旧帧号覆盖新标注。使用同一数据的旧版程序或外部写入程序不受该机制协调。

Slice 导出会先在 `.kirigami-export-*` 临时目录中生成并验证全部新切片。正常编码失败保留旧输出，发布失败时尝试恢复旧输出。多个子任务文件夹的发布并非单次原子操作；强制终止或断电后，应先检查该临时目录和其中的 `recovery.json`，其中可能保留了原有切片的备份。

</details>

## 3D 白模与查看操作

默认显示两台 YAM Ultra 2，基座间距 **46 cm**。白模保留原始 STL 中的孔洞、凹槽、紧固件和夹爪指尖；双臂共 **299,312 个三角面**，网格常驻 GPU 显存，播放时更新关节姿态。光照使用表面法线，并请求 4× 抗锯齿。

| 操作 | 效果 |
| --- | --- |
| `White model` / `Arm colors` | 切换白模或模型配色 |
| 鼠标拖动 3D 区域 | 旋转视角 |
| 滚轮 | 缩放 |
| `Reset` | 恢复默认视角 |
| `Expand` 或双击 3D 区域 | 打开与主界面同步的大视图 |

显示设置不影响记录的关节状态或导出数据。每条机械臂的状态是 7 维：六个关节位置，加夹爪行程。

URDF、MJCF 和 STL 位于 `models/yam_ultra/v2/`，来自 I2RT Robotics，许可证见 [NOTICE](models/yam_ultra/NOTICE.md) 和 [MIT license](models/yam_ultra/LICENSE-i2rt)。需要使用另一份 YAM Ultra 2 模型时，可在启动前设置：

```bash
export KIRIGAMI_YAM_ULTRA_DIR=/path/to/yam_ultra/v2
./run_kirigami.sh
```

## 快捷键

| 快捷键 / 操作 | 功能 |
| --- | --- |
| `Space` | 播放 / 暂停 |
| `←` / `→` | 前后移动一帧 |
| `Shift+←` / `Shift+→` | 按估算帧率前后跳转约 1 秒 |
| `I` / `O` | 设置 Trim 的首个 / 最后一个保留帧 |
| `M` | 添加分界点，重复添加不会删除 |
| `Delete` / `Backspace` | 删除离播放位置最近的分界点 |
| 右键 / 双击时间线 | 在该帧添加或移除分界点 |
| `Ctrl+O` | 打开文件夹 |

精确定位建议用 `FRAME` 输入框或逐帧移动。编辑名称、帧号时，先完成输入，再操作时间线按钮或快捷键。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `conda` | 先 `source` 正确安装目录中的 `etc/profile.d/conda.sh`；尚未安装 Conda 时先完成安装 |
| 提示 `kirigami` 环境不存在或缺少依赖 | 在仓库目录运行 `./setup_env.sh` |
| Qt 报 XCB 动态库缺失 | 重新运行安装脚本；优先通过 `run_kirigami.sh` 启动，以加载环境中的运行库 |
| 提示没有桌面、无法连接 DISPLAY | 在已登录主机图形桌面的终端运行；不要随意填写 DISPLAY 来绕过错误 |
| 继承了 `QT_QPA_PLATFORM=offscreen` 等设置 | 执行 `unset QT_QPA_PLATFORM`，再从桌面终端运行启动脚本 |
| `--check` 的窗口自动关闭 | 属于预期行为；检查结束会退出，正常使用请去掉 `--check` |
| `Trim source…` 按钮不可用 | 先导入 episode 并选择非空保留范围，且至少去掉一帧；保留整段时无需 Trim |
| 在 FRAME 输入框里按 I / O 没反应 | 先点击 Keep in / Keep out 按钮，或移出输入框后再按快捷键 |
| 导出提示 `Missing subtask` | 给每个勾选 Export 的分段填名字，或取消该段的 Export |
| Trim 提示不支持某个数据文件 | 该文件需要明确的同步裁剪规则；保留它并适配格式后再写回 |
| 改过标签后显示未导出 | 修改会使旧导出状态失效，重新导出即可更新切片 |

## 开发与回归检查

在仓库根目录、已激活 `kirigami` 环境的终端中运行：

```bash
python -m unittest tests.test_core tests.test_safety tests.test_robot_meshes tests.test_trimming -v
```

GUI 与 GPU 测试需要主机真实桌面：

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  KIRIGAMI_DESKTOP_TESTS=1 python -m unittest tests.test_gui tests.test_gpu -v
```

这些回归测试主要使用临时合成 episode。真实数据加载测试默认可以跳过；需要启用时，通过 `KIRIGAMI_TEST_EPISODE` 指定符合测试条件的真实样本路径。
