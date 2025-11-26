import random


class ConlluSentence:
    """
    A sentence object containing tokens = [(id, head, form), ...]
    """

    def __init__(self, tokens):
        self.tokens = tokens

    def dependency_length(self):
        """Calculate total dependency length and all arcs of the sentence"""
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
        Completely randomize word order (destroy all structure)
        Strategy 1: Complete Randomization
        Used to test information locality hypothesis
        """
        if seed is not None:
            random.seed(seed)

        forms = [t[2] for t in self.tokens]
        random.shuffle(forms)

        # Build pseudo-sentence: ID unchanged, HEAD=-1 (no structure), FORM=shuffled words
        shuffled_tokens = [
            (i + 1, -1, forms[i]) for i in range(len(forms))
        ]

        return ConlluSentence(shuffled_tokens)
    
    def shuffle_preserve_deps(self, seed=None):
        """
        Preserve dependency relations but randomize positions
        Strategy 2: Dependency-Preserving Randomization
        
        Inspired by the shuffle_tree() method in dependencies.py
        Preserves all dependency relations (governor-dependent pairs), only changes linear positions
        Used to test dependency length minimization hypothesis
        
        Args:
            seed: Random seed
        Returns:
            ConlluSentence: Sentence object with preserved dependencies but shuffled positions
        """
        if seed is not None:
            random.seed(seed)
        
        n = len(self.tokens)
        
        # Create source index list [1, 2, 3, ..., n]
        src_idx = list(range(1, n + 1))
        
        # Create target index (shuffled)
        tgt_idx = src_idx.copy()
        random.shuffle(tgt_idx)
        
        # Create index mapping dictionary {old_pos: new_pos}
        dmap = dict(zip(src_idx, tgt_idx))
        
        # Update dependency relations: map head indices to new positions
        new_tokens = []
        for tid, head, form in self.tokens:
            new_tid = dmap[tid]
            # If head is 0 (root), keep it as 0; otherwise map to new position
            new_head = dmap[head] if head != 0 else 0
            new_tokens.append((new_tid, new_head, form))
        
        # Sort by new tid
        new_tokens.sort(key=lambda x: x[0])
        
        return ConlluSentence(new_tokens)

    def to_text(self):
        """
        Return plain text form of the sentence (word sequence)
        Returns:
            str: Space-separated word sequence
        """
        return ' '.join([form for _, _, form in self.tokens])
    
    def get_word_list(self):
        """
        Return word list of the sentence
        Returns:
            list: List of words
        """
        return [form for _, _, form in self.tokens]

    def __len__(self):
        return len(self.tokens)
    
    def __repr__(self):
        """
        Return readable string representation of the object
        """
        text = self.to_text()
        # Truncate if sentence is too long
        if len(text) > 50:
            text = text[:47] + "..."
        return f"ConlluSentence({len(self.tokens)} words: '{text}')"


class ConlluParser:
    """
    Parse complete CoNLL-U text string
    """

    def parse(self, text):
        sentences = []
        current = []

        for line in text.split("\n"):
            line = line.strip()

            if not line:  # Empty line: end of sentence
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
                continue  # Skip multiword tokens

            tid = int(parts[0])
            form = parts[1]
            head = int(parts[6])

            current.append((tid, head, form))

        if current:
            sentences.append(ConlluSentence(current))

        return sentences


class DependencyAnalyzer:
    """
    Provides statistical and visualization functions
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
        Plot dependency length distribution of sentences:
        - Histogram
        - KDE curve
        - Scatter plot (sentence length vs sentence index)
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 5))

        # Left plot: Histogram + KDE
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

        # Right plot: Scatter plot of dependency length per sentence
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(totals)), totals, s=12)
        plt.title("Sentence-level Dependency Lengths")
        plt.xlabel("Sentence Index")
        plt.ylabel("Dependency Length")

        plt.tight_layout()
        plt.show()


def split_dataset(sentences, train_ratio=0.8, seed=42):
    """
    Split sentence list into training and test sets
    Args:
        sentences (list): List of ConlluSentence objects
        train_ratio (float): Training set ratio, default 0.8
        seed (int): Random seed
    Returns:
        tuple: (train_sentences, test_sentences)
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle sentence order
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_sentences = shuffled[:split_idx]
    test_sentences = shuffled[split_idx:]
    
    return train_sentences, test_sentences


def generate_natural_corpus(corpus_text, max_sentences=None):
    """
    Generate natural corpus from read CoNLL-U text (preserves original dependency structure)
    
    Args:
        corpus_text (str): CoNLL-U format text content that has been read
        max_sentences (int): Maximum number of sentences to use, None means use all
    
    Returns:
        list: List of ConlluSentence objects with preserved original dependency relations
    
    """
    parser = ConlluParser()
    
    # Parse CoNLL-U text
    sentences = parser.parse(corpus_text)
    
    # Truncate if max_sentences is specified
    if max_sentences is not None and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    

    
    return sentences


def generate_complete_random_corpus(corpus_text, max_sentences=None, seed=42):
    """
    Generate completely randomized corpus from read CoNLL-U text (destroys all structure)
    Strategy 1: Test information locality hypothesis
    
    Args:
        corpus_text (str): CoNLL-U format text content that has been read
        max_sentences (int): Maximum number of sentences to use
        seed (int): Random seed to ensure reproducibility
    
    Returns:
        list: List of completely randomized ConlluSentence objects
    
    """
    parser = ConlluParser()
    
    # Parse CoNLL-U text
    sentences = parser.parse(corpus_text)
    
    # Truncate if max_sentences is specified
    if max_sentences is not None and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    
    # Completely randomize each sentence
    shuffled_sentences = [sent.shuffle(seed=seed) for sent in sentences]

    
    return shuffled_sentences


def generate_dep_preserved_corpus(corpus_text, max_sentences=None, seed=42):
    """
    Generate corpus with preserved dependencies but randomized positions from read CoNLL-U text
    Strategy 2: Test dependency length minimization hypothesis
    
    Preserves all dependency relations (governor-dependent pairs), only changes linear positions
    
    Args:
        corpus_text (str): CoNLL-U format text content that has been read
        max_sentences (int): Maximum number of sentences to use
        seed (int): Random seed to ensure reproducibility
    
    Returns:
        list: List of ConlluSentence objects with preserved dependencies but shuffled positions
    
    """
    parser = ConlluParser()
    
    # Parse CoNLL-U text
    sentences = parser.parse(corpus_text)
    
    # Truncate if max_sentences is specified
    if max_sentences is not None and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    
    # Randomize each sentence while preserving dependency relations
    dep_shuffled_sentences = [sent.shuffle_preserve_deps(seed=seed) for sent in sentences]
    
    # Verify dependency length change
    orig_dl, _ = sentences[0].dependency_length()
    shuf_dl, _ = dep_shuffled_sentences[0].dependency_length()
    
    return dep_shuffled_sentences
