"""Seed data factory for the AAD knowledge graph.

Creates a rich demo graph with GPU-related, NVIDIA-triangle, and
distractor nodes to test traversal precision and discrimination.
"""

from aad.models import Node, Association
from aad.embedder import Embedder
from aad.store import AADStore


def create_seed_nodes(embedder: Embedder) -> list[Node]:
    """Create seed nodes with embeddings and bidirectional associations.

    Categories:
      - Core triangle: 黄仁勋, NVIDIA, GPU
      - GPU chain: 图形渲染, 深度学习, 人工智能, GeForce, RTX, 光线追踪, CUDA, 数据中心, 比特币
      - Competitor: AMD
      - Distractors: 苹果, 特斯拉, Python, 量子计算
    """
    nodes_data = [
        # ── Core triangle ──
        {
            "name": "黄仁勋",
            "content": (
                "黄仁勋 (Jensen Huang) 是 NVIDIA 公司的联合创始人、总裁兼首席执行官。"
                "他于 1963 年出生于台湾，1993 年联合创立了 NVIDIA，"
                "带领公司成为 GPU 计算和 AI 加速领域的主导力量。"
                "他以标志性的皮夹克和在加速计算领域的远见领导力而闻名。"
                "黄仁勋大力推动了 CUDA 平台的发展，将 GPU 从游戏显卡转变为通用计算引擎。"
            ),
        },
        {
            "name": "NVIDIA",
            "content": (
                "NVIDIA 公司是一家美国跨国技术公司，由黄仁勋（Jensen Huang）、"
                "Chris Malachowsky 和 Curtis Priem 于 1993 年创立。"
                "总部位于加利福尼亚州圣克拉拉。"
                "NVIDIA 设计 GPU 用于游戏、专业可视化、数据中心和 AI。"
                "其 CUDA 平台是 GPU 加速计算的事实标准。"
                "旗下拥有 GeForce（消费级）、RTX（光线追踪）、"
                "Quadro（专业可视化）、Tesla（数据中心）等产品线。"
                "2023 年市值突破万亿美元，成为 AI 芯片霸主。"
            ),
        },
        {
            "name": "GPU",
            "content": (
                "GPU（图形处理单元）是一种专用并行处理器。"
                "最初为图形渲染设计，现广泛用于通用计算（GPGPU）。"
                "应用领域包括 AI 训练、科学模拟、加密货币挖矿、视频编解码等。"
                "NVIDIA 于 1999 年推出 GeForce 256，被公认为世界上第一款 GPU。"
                "现代 GPU 拥有数千个计算核心，可同时处理大量并行任务。"
                "主要制造商包括 NVIDIA 和 AMD。"
            ),
        },
        # ── GPU 相关链 ──
        {
            "name": "图形渲染",
            "content": (
                "图形渲染是将 3D 场景转换为 2D 图像的计算过程。"
                "包括光栅化渲染和光线追踪渲染两大类。"
                "GPU 从诞生之初就是为了加速图形渲染而设计的。"
                "实时渲染广泛用于电子游戏、虚拟现实和数字孪生。"
                "离线渲染用于电影特效和建筑可视化。"
            ),
        },
        {
            "name": "深度学习",
            "content": (
                "深度学习是机器学习的一个分支，使用多层神经网络从数据中学习。"
                "训练深度神经网络需要大量并行矩阵运算，GPU 是其核心硬件加速器。"
                "NVIDIA 的 CUDA 平台和 cuDNN 库是深度学习训练的事实标准。"
                "主要框架包括 PyTorch、TensorFlow，均支持 GPU 加速。"
                "深度学习在图像识别、自然语言处理、自动驾驶等领域取得突破性进展。"
            ),
        },
        {
            "name": "人工智能",
            "content": (
                "人工智能（AI）是模拟人类智能的计算机科学分支。"
                "包括机器学习、深度学习、自然语言处理、计算机视觉等子领域。"
                "GPU 并行计算能力是 AI 革命的重要硬件基础。"
                "NVIDIA 的数据中心 GPU 是训练大语言模型的核心硬件。"
                "ChatGPT、Claude 等大模型均依赖数万块 GPU 进行训练。"
            ),
        },
        {
            "name": "CUDA",
            "content": (
                "CUDA（Compute Unified Device Architecture）是 NVIDIA 开发的并行计算平台和编程模型。"
                "于 2006 年发布，允许开发者使用 C/C++ 直接编程 GPU。"
                "CUDA 将 GPU 从单纯的图形处理器转变为通用并行计算引擎。"
                "cuDNN、cuBLAS 等 CUDA 库是深度学习框架的底层依赖。"
                "CUDA 生态是 NVIDIA 最重要的护城河，AMD 的 ROCm 目前难以匹敌。"
                "黄仁勋曾说 CUDA 是 NVIDIA 投入最大的长期赌注。"
            ),
        },
        {
            "name": "GeForce",
            "content": (
                "GeForce 是 NVIDIA 面向消费级市场的 GPU 品牌。"
                "1999 年推出首款 GeForce 256，定义了 GPU 的概念。"
                "GeForce 显卡主要用于游戏，也广泛用于内容创作和 AI 推理。"
                "2020 年推出的 GeForce RTX 30 系列采用 Ampere 架构，性能大幅提升。"
                "GeForce Experience 是配套的驱动优化和游戏录制软件。"
            ),
        },
        {
            "name": "RTX",
            "content": (
                "RTX 是 NVIDIA 的光线追踪技术品牌，于 2018 年随 Turing 架构推出。"
                "RTX 系列 GPU 内置专用 RT Core（光线追踪核心）和 Tensor Core（张量核心）。"
                "支持实时光线追踪和 DLSS（深度学习超采样）技术。"
                "RTX 技术大幅提升了游戏画面的光影真实感。"
                "RTX 也用于专业可视化、建筑渲染和电影制作。"
            ),
        },
        {
            "name": "光线追踪",
            "content": (
                "光线追踪是一种通过模拟光线物理行为来生成逼真图像的渲染技术。"
                "传统上用于电影特效的离线渲染，计算量极大。"
                "NVIDIA RTX 系列 GPU 首次实现了消费级的实时光线追踪。"
                "光线追踪需要大量并行计算，GPU 是其天然加速器。"
                "在游戏、建筑可视化和虚拟制作中广泛应用。"
            ),
        },
        {
            "name": "数据中心",
            "content": (
                "数据中心是集中存储、处理和分发数据的设施。"
                "现代数据中心大量使用 GPU 进行 AI 训练和推理加速。"
                "NVIDIA 的数据中心业务（A100、H100 等）是其最大收入来源。"
                "云计算厂商（AWS、Azure、GCP）均提供 GPU 云实例。"
                "AI 大模型的爆发式增长推动了对 GPU 数据中心的巨大需求。"
            ),
        },
        {
            "name": "比特币",
            "content": (
                "比特币（Bitcoin）是 2009 年诞生的去中心化数字货币。"
                "比特币挖矿通过工作量证明（PoW）机制维护网络安全。"
                "早期挖矿使用 CPU，后来 GPU 因其并行计算优势成为主流挖矿硬件。"
                "随着 ASIC 矿机的出现，GPU 挖矿比特币已不经济。"
                "但以太坊等其他加密货币曾长期使用 GPU 挖矿。"
            ),
        },
        # ── 竞争对手 ──
        {
            "name": "AMD",
            "content": (
                "AMD（Advanced Micro Devices）是一家美国半导体公司，"
                "总部位于加利福尼亚州圣克拉拉，与 NVIDIA 同城。"
                "AMD 同时生产 CPU（Ryzen 系列）和 GPU（Radeon 系列）。"
                "其 Radeon GPU 是 NVIDIA GeForce 的主要竞争对手。"
                "AMD 于 2022 年收购了 FPGA 巨头赛灵思（Xilinx）。"
                "Lisa Su（苏姿丰）自 2014 年起担任 AMD CEO，带领公司复兴。"
                "苏姿丰与黄仁勋是远房亲戚，均出生于台湾。"
            ),
        },
        # ── 干扰项（无关联或弱关联）──
        {
            "name": "苹果",
            "content": (
                "苹果公司（Apple Inc.）是一家美国消费电子公司，由乔布斯等人于 1976 年创立。"
                "总部位于加利福尼亚州库比蒂诺。"
                "主要产品包括 iPhone、iPad、Mac、Apple Watch 和 Vision Pro。"
                "苹果自研 M 系列芯片（M1-M4）采用 ARM 架构，性能优异。"
                "苹果使用自家 Metal API 进行图形加速，不支持 NVIDIA CUDA。"
            ),
        },
        {
            "name": "特斯拉",
            "content": (
                "特斯拉（Tesla）是一家美国电动汽车和清洁能源公司。"
                "由 Elon Musk 等人于 2003 年创立，总部位于得克萨斯州奥斯汀。"
                "主要产品包括 Model 3、Model Y、Cybertruck 等电动汽车。"
                "特斯拉自研 FSD（全自动驾驶）芯片用于自动驾驶 AI 推理，"
                "但其自动驾驶芯片并非 GPU，而是专用神经网络加速器。"
            ),
        },
        {
            "name": "Python",
            "content": (
                "Python 是一种解释型高级编程语言，由 Guido van Rossum 于 1991 年发布。"
                "以简洁易读的语法闻名，广泛用于 Web 开发、数据科学和 AI 开发。"
                "Python 是深度学习框架 PyTorch 和 TensorFlow 的主要编程语言。"
                "NumPy、Pandas 等科学计算库大量使用 Python。"
                "Python 本身与 GPU 硬件无关，但可以通过 CUDA Python 调用 GPU。"
            ),
        },
        {
            "name": "量子计算",
            "content": (
                "量子计算是利用量子力学原理进行信息处理的新型计算范式。"
                "使用量子比特（qubit）而非经典比特，具有叠加态和纠缠等特性。"
                "量子计算在密码学、药物发现和材料科学等领域有潜在应用。"
                "IBM、Google、微软等公司正在研发量子计算机。"
                "量子计算与 GPU 属于不同的计算范式，目前无直接关联。"
                "但某些量子模拟算法可以使用 GPU 进行经典近似加速。"
            ),
        },
        # ── 人物家族图谱（测试多跳遍历）──
        {
            "name": "李明",
            "content": (
                "李明，32 岁，软件工程师，在 NVIDIA 上海研发中心工作，"
                "负责 CUDA 驱动开发。毕业于上海交通大学计算机系。"
                "爱好摄影和跑步，周末经常带家人去公园。"
            ),
        },
        {
            "name": "李建国",
            "content": (
                "李建国，62 岁，李明的父亲，退休中学数学教师。"
                "性格严谨，对子女教育非常重视。住在杭州老家。"
            ),
        },
        {
            "name": "王秀兰",
            "content": (
                "王秀兰，60 岁，李明的母亲，退休护士。"
                "热心肠，邻里关系很好，烧得一手好菜。"
                "经常念叨让李明多回家吃饭。"
            ),
        },
        {
            "name": "张晓雯",
            "content": (
                "张晓雯，30 岁，李明的妻子，UI/UX 设计师。"
                "在一家互联网公司工作，擅长插画。"
                "和李明是大学同学，毕业后结婚。"
            ),
        },
        {
            "name": "李小天",
            "content": (
                "李小天，7 岁，李明的儿子，小学一年级。"
                "喜欢乐高和恐龙，梦想当宇航员。"
                "最近迷上了编程启蒙课。"
            ),
        },
        {
            "name": "李小月",
            "content": (
                "李小月，4 岁，李明的女儿，幼儿园中班。"
                "活泼可爱，喜欢画画和跳舞。"
                "是全家人的开心果。"
            ),
        },
        {
            "name": "李亮",
            "content": (
                "李亮，28 岁，李明的弟弟，三甲医院外科医生。"
                "性格沉稳，医术精湛。工作繁忙但每周都会给父母打电话。"
            ),
        },
        {
            "name": "李德胜",
            "content": (
                "李德胜，88 岁，李明的爷爷（李建国的父亲）。"
                "参加过抗美援朝的老兵，身体硬朗。"
                "住在杭州老家，每天早起打太极。"
            ),
        },
        {
            "name": "王大海",
            "content": (
                "王大海，84 岁，李明的外公（王秀兰的父亲）。"
                "退休前是机械厂工程师，手艺精湛。"
                "喜欢钓鱼和下象棋。"
            ),
        },
        {
            "name": "王建国",
            "content": (
                "王建国，58 岁，李明的舅舅（王秀兰的哥哥）。"
                "开了一家小型物流公司，生意不错。"
                "逢年过节总是组织家族聚会。"
            ),
        },
        {
            "name": "张伟",
            "content": (
                "张伟，63 岁，李明的岳父（张晓雯的父亲）。"
                "退休公务员，喜欢书法和养花。"
                "对李明这个女婿很满意。"
            ),
        },
        {
            "name": "刘芳",
            "content": (
                "刘芳，61 岁，李明的岳母（张晓雯的母亲）。"
                "退休会计，做事细致。"
                "擅长编织，给孙子孙女织了很多毛衣。"
            ),
        },
        {
            "name": "李小花",
            "content": (
                "李小花，5 岁，李明的侄女（李亮的女儿）。"
                "聪明伶俐，喜欢缠着李小天哥哥玩。"
                "在幼儿园是班长。"
            ),
        },
    ]

    # Generate embeddings
    texts = [d["name"] + " " + d["content"] for d in nodes_data]
    vectors = embedder.embed_batch(texts)

    # Build nodes
    nodes: dict[str, Node] = {}
    for data, vec in zip(nodes_data, vectors):
        nodes[data["name"]] = Node(
            name=data["name"],
            content=data["content"],
            vector=vec,
        )

    n = nodes  # shorthand

    # ── Helper ──
    def link(a: str, b: str, reason_a: str, reason_b: str) -> None:
        n[a].associations.append(Association(vector=n[b].vector, reason=reason_a))
        n[b].associations.append(Association(vector=n[a].vector, reason=reason_b))

    link("黄仁勋", "NVIDIA", "联合创立了 NVIDIA（1993 年）", "由黄仁勋联合创立")
    link("NVIDIA", "GPU", "发明了 GPU（GeForce 256，1999 年）", "GPU 由 NVIDIA 发明")
    link("黄仁勋", "GPU", "领导了发明 GPU 的公司", "GPU 由黄仁勋领导的 NVIDIA 发明")

    # ── GPU 链 ──
    link("GPU", "图形渲染", "GPU 最初为图形渲染设计", "图形渲染由 GPU 硬件加速")
    link("GPU", "深度学习", "GPU 是深度学习训练的核心加速器", "深度学习依赖 GPU 并行计算")
    link("GPU", "GeForce", "GeForce 是面向消费者的 GPU 品牌", "GeForce 是 GPU 的一个产品系列")
    link("GPU", "比特币", "GPU 曾广泛用于加密货币挖矿", "比特币挖矿早期大量使用 GPU")
    link("GPU", "RTX", "RTX 是 NVIDIA 的高端 GPU 产品线", "RTX 是 GPU 光线追踪的标杆")

    link("NVIDIA", "CUDA", "开发了 CUDA 并行计算平台", "CUDA 是 NVIDIA 的核心软件生态")
    link("NVIDIA", "GeForce", "GeForce 是 NVIDIA 的消费级品牌", "GeForce 由 NVIDIA 推出")
    link("NVIDIA", "数据中心", "数据中心 GPU 是 NVIDIA 最大收入来源", "NVIDIA 是数据中心 GPU 的主要供应商")
    link("NVIDIA", "RTX", "RTX 是 NVIDIA 的光线追踪技术品牌", "RTX 技术由 NVIDIA 首创")
    link("NVIDIA", "AMD", "NVIDIA 与 AMD 是 GPU 市场主要竞争对手", "AMD 是 NVIDIA 的主要竞争对手")

    link("黄仁勋", "CUDA", "大力推动 CUDA 平台发展", "CUDA 由黄仁勋主导的战略决策")
    link("黄仁勋", "AMD", "黄仁勋的 NVIDIA 与 AMD 竞争", "AMD CEO 苏姿丰与黄仁勋是远房亲戚")

    link("CUDA", "深度学习", "CUDA 是深度学习框架的底层计算平台", "深度学习训练依赖 CUDA 加速")
    link("深度学习", "人工智能", "深度学习是 AI 的核心技术分支", "AI 依赖深度学习的突破")
    link("图形渲染", "光线追踪", "光线追踪是高级图形渲染技术", "光线追踪属于图形渲染的一种方法")
    link("RTX", "光线追踪", "RTX GPU 支持实时光线追踪", "光线追踪由 RTX 硬件加速")
    link("AMD", "GPU", "AMD 也生产 GPU（Radeon 系列）", "GPU 制造商包括 AMD")
    link("数据中心", "人工智能", "数据中心提供 AI 训练的算力基础设施", "AI 大模型训练依赖数据中心 GPU 集群")

    # ── 李明家族图谱（测试多跳）──
    link("李明", "李建国", "李建国的儿子", "李明的父亲")
    link("李明", "王秀兰", "王秀兰的儿子", "李明的母亲")
    link("李明", "张晓雯", "张晓雯的丈夫", "李明的妻子")
    link("李明", "李小天", "李小天的爸爸", "李明的儿子")
    link("李明", "李小月", "李小月的爸爸", "李明的女儿")
    link("李明", "李亮", "李亮的哥哥", "李明的弟弟")

    link("李建国", "王秀兰", "王秀兰的丈夫", "李建国的妻子")
    link("李建国", "李德胜", "李德胜的儿子", "李建国的父亲")
    link("李建国", "李亮", "李亮的父亲", "李建国的二儿子")

    link("王秀兰", "王大海", "王大海的女儿", "王秀兰的父亲")
    link("王秀兰", "王建国", "王建国的妹妹", "王秀兰的哥哥")

    link("张晓雯", "李小天", "李小天的妈妈", "张晓雯的儿子")
    link("张晓雯", "李小月", "李小月的妈妈", "张晓雯的女儿")
    link("张晓雯", "张伟", "张伟的女儿", "张晓雯的父亲")
    link("张晓雯", "刘芳", "刘芳的女儿", "张晓雯的母亲")

    link("张伟", "刘芳", "刘芳的丈夫", "张伟的妻子")

    link("李亮", "李小花", "李小花的爸爸", "李亮的女儿")

    link("李小天", "李小月", "李小月的哥哥", "李小天的妹妹")

    # ── 桥接：李明 → NVIDIA → 现有图谱 ──
    link("李明", "NVIDIA", "在 NVIDIA 上海研发中心工作", "员工李明，负责 CUDA 驱动开发")

    return list(nodes.values())


def seed_store(store: AADStore, embedder: Embedder) -> int:
    """Populate an AADStore with seed nodes. Returns count of nodes added."""
    nodes = create_seed_nodes(embedder)
    for node in nodes:
        store.put(node)
    return len(nodes)
