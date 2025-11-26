import random


class ConlluSentence:
    """
    一个句子对象，包含 tokens = [(id, head, form), ...]
    """

    def __init__(self, tokens):
        self.tokens = tokens

    def dependency_length(self):
        """计算句子的总依存长度和所有 arcs"""
        total = 0
        arcs = []
        for tid, head, form in self.tokens:
            if head == 0:
                continue
            dist = abs(tid - head)
            total += dist
            arcs.append((form, tid, head, dist))
        return total, arcs

    def shuffle(self, seed=None):
        """
        随机打乱句子顺序（不保留结构）
        用于生成 counterfactual baseline
        """
        if seed is not None:
            random.seed(seed)

        forms = [t[2] for t in self.tokens]
        random.shuffle(forms)

        # 构建伪句子：ID 不变，HEAD=-1（无结构），FORM=打乱后的词
        shuffled_tokens = [
            (i + 1, -1, forms[i]) for i in range(len(forms))
        ]

        return ConlluSentence(shuffled_tokens)

    def to_text(self):
        """
        返回句子的纯文本形式（词序列）
        Returns:
            str: 空格分隔的词序列
        """
        return ' '.join([form for _, _, form in self.tokens])
    
    def get_word_list(self):
        """
        返回句子的词列表
        Returns:
            list: 词的列表
        """
        return [form for _, _, form in self.tokens]

    def __len__(self):
        return len(self.tokens)


class ConlluParser:
    """
    解析完整 CoNLL-U 文本字符串
    """

    def parse(self, text):
        sentences = []
        current = []

        for line in text.split("\n"):
            line = line.strip()

            if not line:  # 空行：句子结束
                if current:
                    sentences.append(ConlluSentence(current))
                    current = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 7:
                continue

            token_id = parts[0]
            if "-" in token_id or "." in token_id:
                continue  # 跳过 multiword token

            tid = int(parts[0])
            form = parts[1]
            head = int(parts[6])

            current.append((tid, head, form))

        if current:
            sentences.append(ConlluSentence(current))

        return sentences


class DependencyAnalyzer:
    """
    提供统计与可视化功能
    """

    def total_and_average_length(self, sentences):
        totals = []
        for s in sentences:
            total, _ = s.dependency_length()
            totals.append(total)
        avg = sum(totals) / len(totals)
        return totals, avg

    def plot_dependency_lengths(self, totals, title="Dependency Length Distribution"):
        """
        绘制句子依存长度分布图：
        - 直方图
        - KDE 曲线
        - 散点图（句长 vs 句子索引）
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 5))

        # 左图：直方图 + KDE
        plt.subplot(1, 2, 1)
        plt.hist(totals, bins=30, density=True, alpha=0.6)
        try:
            from scipy.stats import gaussian_kde
            import numpy as np
            kde = gaussian_kde(totals)
            xs = np.linspace(min(totals), max(totals), 200)
            plt.plot(xs, kde(xs))
        except:
            pass
        plt.title(title)
        plt.xlabel("Dependency Length")
        plt.ylabel("Density")

        # 右图：每句 DL 的散点图
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(totals)), totals, s=12)
        plt.title("Sentence-level Dependency Lengths")
        plt.xlabel("Sentence Index")
        plt.ylabel("Dependency Length")

        plt.tight_layout()
        plt.show()


def split_dataset(sentences, train_ratio=0.8, seed=42):
    """
    将句子列表分割为训练集和测试集
    Args:
        sentences (list): ConlluSentence 对象列表
        train_ratio (float): 训练集比例，默认 0.8
        seed (int): 随机种子
    Returns:
        tuple: (train_sentences, test_sentences)
    """
    if seed is not None:
        random.seed(seed)
    
    # 打乱句子顺序
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    
    # 分割
    split_idx = int(len(shuffled) * train_ratio)
    train_sentences = shuffled[:split_idx]
    test_sentences = shuffled[split_idx:]
    
    return train_sentences, test_sentences
