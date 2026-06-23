#!/usr/bin/env python3
import os
import sys
import json
import re
import time
import random
import warnings
import glob
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from languages import pdf_to_text
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import os
os.makedirs("multi_bot_results", exist_ok=True)
print("Папка создана:", os.path.abspath("multi_bot_results"))

# Бот 1: Марковские цепи 
class MarkovChainBot:
    def __init__(self, n=3):
        self.n = n
        self.chain = defaultdict(list)
        self.vocab = set()

    def train(self, text):
        words = text.split()
        self.vocab.update(words)
        for i in range(len(words) - self.n):
            key = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            self.chain[key].append(next_word)
        print(f"Марковская цепь: {len(self.chain)} состояний, {len(self.vocab)} уникальных слов")

    def generate(self, seed_text, length=3000, temperature=0.7):
        seed_words = seed_text.split()
        if len(seed_words) < self.n:
            seed_words = (['the'] * (self.n - len(seed_words))) + seed_words
        result = seed_words[-self.n:]
        for _ in range(length):
            key = tuple(result[-self.n:])
            if key in self.chain and self.chain[key]:
                choices = self.chain[key]
                if temperature <= 0:
                    next_word = max(set(choices), key=choices.count)
                else:
                    weights = Counter(choices)
                    total = sum(weights.values())
                    probs = {k: (v/total) ** (1/temperature) for k, v in weights.items()}
                    probs_total = sum(probs.values())
                    probs = {k: v/probs_total for k, v in probs.items()}
                    words_list, probabilities = zip(*probs.items())
                    next_word = random.choices(words_list, weights=probabilities, k=1)[0]
                result.append(next_word)
            else:
                if self.vocab:
                    result.append(random.choice(list(self.vocab)))
                break
        return ' '.join(result)

# Бот 2: Улучшенный LSTM
class BetterLSTMBot:
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.3):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def train(self, text, epochs=30, lr=0.003):
        print(f"Обучение BetterLSTM: {epochs} эпох")
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(text) - 1):
            current = text[i]
            next_char = text[i+1]
            self.transitions[current][next_char] += 1
        self.probabilities = {}
        for char, next_chars in self.transitions.items():
            total = sum(next_chars.values())
            self.probabilities[char] = {k: v/total for k, v in next_chars.items()}
          
    def generate(self, start_text, length=3000, temperature=0.8):
        if not hasattr(self, 'probabilities'):
            return start_text
        result = list(start_text)
        for _ in range(length):
            current = result[-1] if result else ' '
            if current in self.probabilities:
                probs = self.probabilities[current]
                chars, probabilities = zip(*probs.items())
                if temperature != 1.0:
                    probabilities = np.array(probabilities) ** (1.0 / temperature)
                    probabilities /= probabilities.sum()
                next_char = np.random.choice(chars, p=probabilities)
                result.append(next_char)
            else:
                if hasattr(self, 'idx_to_char'):
                    result.append(self.idx_to_char.get(random.randint(0, self.vocab_size-1), ' '))
                else:
                    result.append(' ')
        return ''.join(result)

# Бот 3: Простой LSTM
class SimpleBotWrapper:
    def __init__(self):
        self.model = None

    def train(self, text, epochs=20, hidden_size=128):
        self.model = self._create_emulated_model(text)

    def _create_emulated_model(self, text):
        words = text.split()
        self.vocab = set(words)
        self.bigram = defaultdict(Counter)
        for i in range(len(words)-1):
            self.bigram[words[i]][words[i+1]] += 1
        for w in self.bigram:
            total = sum(self.bigram[w].values())
            for nw in self.bigram[w]:
                self.bigram[w][nw] /= total
        return self

    def generate(self, start_text, length=3000, temperature=0.8):
        if not hasattr(self, 'bigram'):
            return start_text
        words = start_text.split()
        if not words:
            words = [random.choice(list(self.vocab))] if self.vocab else ['the']
        result = words[:]
        for _ in range(length):
            current = result[-1]
            if current in self.bigram and self.bigram[current]:
                items = list(self.bigram[current].items())
                words_list, probs = zip(*items)
                probs = np.array(probs) ** (1.0/temperature)
                probs /= probs.sum()
                next_word = np.random.choice(words_list, p=probs)
                result.append(next_word)
            else:
                if self.vocab:
                    result.append(random.choice(list(self.vocab)))
                else:
                    break
        return ' '.join(result)

# Бот 4: Hybrid
class HybridBot:
    def __init__(self, n_gram_order=5, neural_dim=128):
        self.n_gram_order = n_gram_order
        self.neural_dim = neural_dim
        self.ngram_model = defaultdict(Counter)
        self.word_embeddings = {}
        self.lang_params = {}

    def train(self, text, epochs=20, lang='en'):
        print(f"Обучение Hybrid Bot: {epochs} эпох [{lang}]")
        self.lang_params = self._get_language_params(lang)
        words = text.split()
        n_order = self.lang_params.get('n_order', self.n_gram_order)
        for i in range(len(words) - n_order):
            context = tuple(words[i:i+n_order])
            next_word = words[i+n_order]
            self.ngram_model[context][next_word] += 1
        window_size = self.lang_params.get('window_size', 5)
        for i, word in enumerate(words):
            if word not in self.word_embeddings:
                self.word_embeddings[word] = np.zeros(self.neural_dim)
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_word = words[j]
                    if context_word not in self.word_embeddings:
                        self.word_embeddings[context_word] = np.zeros(self.neural_dim)
                    lr = self.lang_params.get('embedding_lr', 0.01)
                    self.word_embeddings[word] += np.random.randn(self.neural_dim) * lr

    def _get_language_params(self, lang):
        params = {
            'en': {'n_order': 4, 'window_size': 5, 'temperature': 0.7, 'embedding_lr': 0.01, 'repetition_penalty': 0.8},
            'ru': {'n_order': 5, 'window_size': 7, 'temperature': 0.85, 'embedding_lr': 0.015, 'repetition_penalty': 0.6},
            'rom': {'n_order': 6, 'window_size': 8, 'temperature': 0.8, 'embedding_lr': 0.02, 'repetition_penalty': 0.5}
        }
        return params.get(lang, params['en'])

    def generate(self, seed_text, length=1500, temperature=None):
        if temperature is None:
            temperature = self.lang_params.get('temperature', 0.8)
        words = seed_text.split()
        n_order = self.lang_params.get('n_order', self.n_gram_order)
        if len(words) < n_order:
            filler = ['the'] * 10 if self.lang_params.get('temperature', 0.8) < 0.8 else ['и'] * 10
            words = filler[:max(0, n_order - len(words))] + words
        recent_words = []
        for _ in range(length):
            context = tuple(words[-n_order:])
            next_word = None
            if context in self.ngram_model and self.ngram_model[context]:
                candidates = self.ngram_model[context]
                repetition_penalty = self.lang_params.get('repetition_penalty', 0.7)
                adjusted_candidates = {}
                for word, count in candidates.items():
                    penalty = repetition_penalty if word in recent_words[-10:] else 1.0
                    adjusted_candidates[word] = count * penalty
                total = sum(adjusted_candidates.values())
                if total > 0:
                    if temperature > 0:
                        probs = {k: (v/total)**(1/temperature) for k, v in adjusted_candidates.items()}
                        probs_sum = sum(probs.values())
                        probs = {k: v/probs_sum for k, v in probs.items()}
                        next_words, probabilities = zip(*probs.items())
                        next_word = np.random.choice(next_words, p=probabilities)
                    else:
                        next_word = max(adjusted_candidates, key=adjusted_candidates.get)
            if next_word is None and self.word_embeddings:
                if words[-1] in self.word_embeddings:
                    target_embedding = self.word_embeddings[words[-1]]
                    similarities = {}
                    for word, embedding in list(self.word_embeddings.items())[:1000]:
                        if word not in recent_words[-5:]:
                            sim = np.dot(target_embedding, embedding) / (
                                np.linalg.norm(target_embedding) * np.linalg.norm(embedding) + 1e-8)
                            similarities[word] = sim
                    if similarities:
                        words_list = list(similarities.keys())
                        weights = np.array([max(0, v) for v in similarities.values()])
                        if weights.sum() > 0:
                            weights = weights / weights.sum()
                            next_word = np.random.choice(words_list, p=weights)
            if next_word:
                words.append(next_word)
                recent_words.append(next_word)
                if len(recent_words) > 20:
                    recent_words.pop(0)
            else:
                if self.word_embeddings:
                    words.append(random.choice(list(self.word_embeddings.keys())))
                else:
                    break
        return ' '.join(words)

# Бот 5: CNN+Attention
class CNNAttentionBot:
    def __init__(self, vocab_size=10000, embed_dim=256, num_filters=128, kernel_sizes=[3,5,7]):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.transition_matrix = None

    def _build_vocabulary(self, text, max_vocab=10000):
        words = text.split()
        word_counts = Counter(words)
        most_common = word_counts.most_common(max_vocab - 2)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        for i, (word, _) in enumerate(most_common):
            idx = i + 2
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        self.vocab_size = len(self.word_to_idx)

    def _build_attention_weights(self, text, context_size=5):
        words = text.split()
        attention_matrix = defaultdict(lambda: defaultdict(float))
        for i in range(len(words)):
            start = max(0, i - context_size)
            end = min(len(words), i + context_size + 1)
            context_words = words[start:end]
            target_word = words[i]
            for j, context_word in enumerate(context_words):
                if context_word != target_word:
                    distance = abs(j - context_size)
                    attention_weight = 1.0 / (1.0 + distance)
                    attention_matrix[target_word][context_word] += attention_weight
        for word in attention_matrix:
            total = sum(attention_matrix[word].values())
            if total > 0:
                for ctx_word in attention_matrix[word]:
                    attention_matrix[word][ctx_word] /= total
        return attention_matrix

    def train(self, text, epochs=30):
        self._build_vocabulary(text)
        self.attention_weights = self._build_attention_weights(text)
        words = text.split()
        self.transition_matrix = defaultdict(Counter)
        for i in range(len(words) - 1):
            current = words[i]
            next_word = words[i+1]
            self.transition_matrix[current][next_word] += 1
            
    def generate(self, seed_text, length=1500, temperature=0.8):
        words = seed_text.split()
        for _ in range(length):
            current_word = words[-1] if words else ''
            candidates = []
            if current_word in self.transition_matrix:
                transitions = self.transition_matrix[current_word]
                for word, count in transitions.most_common(10):
                    candidates.append((word, count))
            if current_word in self.attention_weights:
                attention = self.attention_weights[current_word]
                for word, weight in sorted(attention.items(), key=lambda x: x[1], reverse=True)[:10]:
                    candidates.append((word, weight * 10))
            if candidates:
                word_scores = defaultdict(float)
                for word, score in candidates:
                    if len(words) < 3 or word != words[-1]:
                        word_scores[word] += score
                if word_scores:
                    words_list = list(word_scores.keys())
                    scores = np.array([word_scores[w] for w in words_list])
                    if temperature > 0:
                        scores = scores ** (1.0 / temperature)
                    if scores.sum() > 0:
                        probs = scores / scores.sum()
                        next_word = np.random.choice(words_list, p=probs)
                        words.append(next_word)
                        continue
            if self.idx_to_word and len(self.idx_to_word) > 2:
                rand_idx = random.randint(2, min(self.vocab_size, len(self.idx_to_word)) - 1)
                words.append(self.idx_to_word.get(rand_idx, 'the'))
        return ' '.join(words)
        
# Бот 6: Advanced N-gram bot
class AdvancedNGramBot:
    def __init__(self, max_n=5):
        self.max_n = max_n
        self.ngrams = defaultdict(lambda: defaultdict(float))
        self.vocabulary = []
        self.lang_params = {}

    def train(self, text, epochs=5, lang='en'):
        print(f"Обучение Advanced N-gram Bot [{lang}]...")
        self.lang_params = self._get_language_params(lang)
        words = text.split()
        self.vocabulary = list(set(words))
        max_n = self.lang_params.get('max_n', self.max_n)
        for n in range(1, max_n + 1):
            for i in range(len(words) - n):
                context = tuple(words[i:i+n])
                next_word = words[i+n]
                self.ngrams[context][next_word] += 1
        for context in self.ngrams:
            total = sum(self.ngrams[context].values())
            if total > 0:
                for word in self.ngrams[context]:
                    self.ngrams[context][word] /= total
        print(f"Обучено {len(self.ngrams)} паттернов")

    def _get_language_params(self, lang):
        return {
            'en': {'max_n': 5, 'temperature': 0.7},
            'ru': {'max_n': 7, 'temperature': 0.9},
            'rom': {'max_n': 6, 'temperature': 0.85}
        }.get(lang, {'max_n': 5, 'temperature': 0.8})

    def generate(self, seed_text, length=1500, temperature=None):
        if temperature is None:
            temperature = self.lang_params.get('temperature', 0.8)
        max_n = self.lang_params.get('max_n', self.max_n)
        words = seed_text.split()
        if len(words) < max_n:
            filler = ['the', 'and', 'of'] * 5
            words = filler[:max_n - len(words)] + words
        for _ in range(length):
            next_word = self._predict(words, temperature, max_n)
            if next_word:
                words.append(next_word)
            elif self.vocabulary:
                words.append(random.choice(self.vocabulary))
        return ' '.join(words)

    def _predict(self, words, temperature, max_n):
        candidates = defaultdict(float)
        for n in range(min(max_n, len(words)), 0, -1):
            context = tuple(words[-n:])
            if context in self.ngrams and self.ngrams[context]:
                for word, prob in self.ngrams[context].items():
                    if word in words[-5:]:
                        prob *= 0.3
                    candidates[word] += prob * (n / max_n)
                if candidates:
                    break
        if not candidates:
            return None
        words_list = list(candidates.keys())
        scores = np.array([candidates[w] for w in words_list])
        if temperature > 0:
            scores = np.exp(np.log(scores + 1e-10) / temperature)
        scores = scores / scores.sum()
        return np.random.choice(words_list, p=scores)

# Бот 7: Statistical LM bot
class StatisticalLMBot:
    def __init__(self):
        self.word_freq = Counter()
        self.bigram_freq = defaultdict(Counter)
        self.trigram_freq = defaultdict(Counter)
        self.sentence_starts = []

    def train(self, text, epochs=10):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 3:
                continue
            self.sentence_starts.append(tuple(words[:3]))
            for word in words:
                self.word_freq[word] += 1
            for i in range(len(words) - 1):
                self.bigram_freq[words[i]][words[i+1]] += 1
            for i in range(len(words) - 2):
                key = (words[i], words[i+1])
                self.trigram_freq[key][words[i+2]] += 1
        self._normalize_probs()

    def _normalize_probs(self):
        total_words = sum(self.word_freq.values())
        for word in self.word_freq:
            self.word_freq[word] /= total_words
        for w1 in self.bigram_freq:
            total = sum(self.bigram_freq[w1].values())
            if total > 0:
                for w2 in self.bigram_freq[w1]:
                    self.bigram_freq[w1][w2] /= total
        for key in self.trigram_freq:
            total = sum(self.trigram_freq[key].values())
            if total > 0:
                for word in self.trigram_freq[key]:
                    self.trigram_freq[key][word] /= total

    def generate(self, seed_text, length=1500, temperature=0.8):
        words = seed_text.split()
        if len(words) < 3:
            if self.sentence_starts:
                words = list(random.choice(self.sentence_starts))
            else:
                words.extend(['the', 'and', 'of'])
        for _ in range(length):
            next_word = self._select_next(words, temperature)
            if next_word:
                words.append(next_word)
                if random.random() < 0.1 and self.sentence_starts:
                    words.extend(random.choice(self.sentence_starts))
            elif self.word_freq:
                words.append(random.choice(list(self.word_freq.keys())))
        return ' '.join(words)

    def _select_next(self, words, temperature):
        candidates = defaultdict(float)
        if len(words) >= 2:
            key = (words[-2], words[-1])
            if key in self.trigram_freq:
                for word, prob in self.trigram_freq[key].items():
                    candidates[word] += prob * 3.0
        if words[-1] in self.bigram_freq:
            for word, prob in self.bigram_freq[words[-1]].items():
                if word not in words[-3:]:
                    candidates[word] += prob * 1.5
        if len(candidates) < 5:
            for word, prob in self.word_freq.most_common(20):
                if word not in words[-5:]:
                    candidates[word] += prob * 0.5
        if not candidates:
            return None
        words_list = list(candidates.keys())
        scores = np.array([candidates[w] for w in words_list])
        if temperature > 0:
            scores = np.exp(np.log(scores + 1e-10) / temperature)
        scores = scores / scores.sum()
        return np.random.choice(words_list, p=scores)

#  Семантические траектории, сетевой анализ, метрики
class SemanticTrajectoryAnalyzer:
    def __init__(self, word_embeddings=None):
        self.sentence_model = None
        self.use_external = False
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.use_external = True
        self.word_embeddings = word_embeddings if word_embeddings else {}

    def _get_sentence_vector(self, sentence):
        if self.use_external and self.sentence_model:
            return self.sentence_model.encode([sentence])[0]
        words = sentence.split()
        vectors = []
        for w in words:
            if w in self.word_embeddings:
                vectors.append(self.word_embeddings[w])
        if vectors:
            return np.mean(vectors, axis=0)
        return np.random.randn(50)

    def get_trajectory(self, text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if not sentences:
            return np.array([])
        embeddings = np.array([self._get_sentence_vector(s) for s in sentences])
        return embeddings

    def compute_trajectory_metrics(self, embeddings):
        n = len(embeddings)
        if n < 2:
            return {'trajectory_length': 0.0, 'mean_speed': 0.0, 'tortuosity': 0.0}
        dists = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)
        length = np.sum(dists)
        mean_speed = np.mean(dists)
        direct_dist = np.linalg.norm(embeddings[-1] - embeddings[0])
        tortuosity = length / direct_dist if direct_dist > 0 else 1.0
        return {
            'trajectory_length': float(length),
            'mean_speed': float(mean_speed),
            'tortuosity': float(tortuosity)
        }

    def dtw_distance(self, emb1, emb2):
        n, m = len(emb1), len(emb2)
        if n == 0 or m == 0:
            return float('inf')
        dtw = np.full((n+1, m+1), np.inf)
        dtw[0, 0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.linalg.norm(emb1[i-1] - emb2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        return dtw[n, m]

    def trajectory_similarity(self, emb_human, emb_bot):
        if len(emb_human) == 0 or len(emb_bot) == 0:
            return 0.0
        dtw_val = self.dtw_distance(emb_human, emb_bot)
        human_len = np.sum(np.linalg.norm(np.diff(emb_human, axis=0), axis=1)) if len(emb_human) > 1 else 0.0
        bot_len = np.sum(np.linalg.norm(np.diff(emb_bot, axis=0), axis=1)) if len(emb_bot) > 1 else 0.0
        avg_len = (human_len + bot_len) / 2.0
        if avg_len == 0:
            return 1.0 if dtw_val == 0 else 0.0
        similarity = np.exp(-dtw_val / avg_len)
        return float(similarity)

class NetworkAnalyzer:
    def __init__(self, lower=True):
        self.lower = lower

    def _tokenize(self, text):
        tokens = text.split()
        if self.lower:
            tokens = [t.lower() for t in tokens]
        return tokens

    def build_graph(self, text):
        tokens = self._tokenize(text)
        edge_counts = Counter()
        out_counts = Counter()
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i+1]
            edge_counts[(a, b)] += 1
            out_counts[a] += 1
        node_degrees = Counter()
        for (a, b), w in edge_counts.items():
            node_degrees[a] += w
            node_degrees[b] += w
        for t in tokens:
            if t not in node_degrees:
                node_degrees[t] = 0
        return tokens, edge_counts, out_counts, node_degrees

    def compute_assortativity(self, edge_counts, node_degrees):
        if not edge_counts:
            return 0.0
        src = []
        tgt = []
        wgt = []
        for (a, b), w in edge_counts.items():
            src.append(node_degrees.get(a, 0))
            tgt.append(node_degrees.get(b, 0))
            wgt.append(w)
        src = np.array(src, dtype=float)
        tgt = np.array(tgt, dtype=float)
        wgt = np.array(wgt, dtype=float)
        mean_src = np.average(src, weights=wgt)
        mean_tgt = np.average(tgt, weights=wgt)
        cov = np.average((src - mean_src) * (tgt - mean_tgt), weights=wgt)
        std_src = np.sqrt(np.average((src - mean_src)**2, weights=wgt))
        std_tgt = np.sqrt(np.average((tgt - mean_tgt)**2, weights=wgt))
        if std_src == 0 or std_tgt == 0:
            return 0.0
        return float(cov / (std_src * std_tgt))

    def compute_local_assortativity(self, edge_counts, node_degrees):
        neighbors = defaultdict(list)
        for (a, b), w in edge_counts.items():
            neighbors[a].append((b, w))
        knn = {}
        for node, neigh_list in neighbors.items():
            total_weight = sum(w for _, w in neigh_list)
            if total_weight > 0:
                avg_deg = sum(node_degrees.get(nb, 0) * w for nb, w in neigh_list) / total_weight
            else:
                avg_deg = 0.0
            knn[node] = avg_deg
        return knn

    def top_hubs(self, node_degrees, topn=10):
        return node_degrees.most_common(topn)

    def compute_transition_distribution(self, out_counts, edge_counts):
        probs = []
        for a, cnt_out in out_counts.items():
            for (src, tgt), w in edge_counts.items():
                if src == a:
                    probs.append(w / cnt_out)
        return probs

    def compute_all(self, text):
        tokens, edge_counts, out_counts, node_degrees = self.build_graph(text)
        assort = self.compute_assortativity(edge_counts, node_degrees)
        knn = self.compute_local_assortativity(edge_counts, node_degrees)
        hubs = self.top_hubs(node_degrees, topn=15)
        trans_probs = self.compute_transition_distribution(out_counts, edge_counts)
        degree_values = list(node_degrees.values())
        mean_degree = np.mean(degree_values) if degree_values else 0.0
        std_degree = np.std(degree_values) if degree_values else 0.0
        degrees_for_knn = [node_degrees[node] for node in knn.keys()]
        knn_for_plot = [knn[node] for node in knn.keys()]
        return {
            'num_nodes': len(node_degrees),
            'num_edges': len(edge_counts),
            'mean_degree': mean_degree,
            'std_degree': std_degree,
            'assortativity': assort,
            'hubs': hubs,
            'degree_distribution': degree_values,
            'knn': knn,
            'knn_mean': np.mean(knn_for_plot) if knn_for_plot else 0.0,
            'degrees_for_knn': degrees_for_knn,
            'knn_for_plot': knn_for_plot,
            'transition_probs': trans_probs,
            'mean_trans_prob': np.mean(trans_probs) if trans_probs else 0.0,
            'std_trans_prob': np.std(trans_probs) if trans_probs else 0.0
        }

class TextMetrics:
    @staticmethod
    def calculate_perplexity(text, n=2):
        words = text.split()
        if len(words) < n + 1:
            return float('inf')
        ngram_counts = defaultdict(Counter)
        context_counts = Counter()
        for i in range(len(words) - n):
            context = tuple(words[i:i+n])
            next_word = words[i+n]
            ngram_counts[context][next_word] += 1
            context_counts[context] += 1
        log_prob_sum = 0
        count = 0
        for i in range(len(words) - n):
            context = tuple(words[i:i+n])
            next_word = words[i+n]
            if context in ngram_counts and context_counts[context] > 0:
                prob = ngram_counts[context][next_word] / context_counts[context]
                if prob > 0:
                    log_prob_sum += np.log2(prob)
                    count += 1
        if count == 0:
            return float('inf')
        avg_log_prob = -log_prob_sum / count
        perplexity = 2 ** avg_log_prob
        return perplexity

    @staticmethod
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return TextMetrics.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    @staticmethod
    def calculate_levenshtein_similarity(text1, text2, sample_size=1000):
        sample1 = text1[:sample_size]
        sample2 = text2[:sample_size]
        distance = TextMetrics.levenshtein_distance(sample1, sample2)
        max_len = max(len(sample1), len(sample2))
        if max_len == 0:
            return 0
        similarity = 1 - (distance / max_len)
        return max(0, similarity)

    @staticmethod
    def calculate_syntactic_complexity(text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if not sentences:
            return 0, {}
        metrics = {
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]),
            'std_sentence_length': np.std([len(s.split()) for s in sentences]),
            'unique_words_ratio': len(set(text.split())) / len(text.split()) if text.split() else 0,
            'punctuation_density': len(re.findall(r'[.,!?;:()\-\'"]', text)) / len(text) if text else 0,
            'num_sentences': len(sentences)
        }
        complexity_score = (
            metrics['avg_sentence_length'] * 0.3 +
            metrics['std_sentence_length'] * 0.2 +
            metrics['unique_words_ratio'] * 100 * 0.3 +
            metrics['punctuation_density'] * 1000 * 0.2
        )
        return complexity_score, metrics

    @staticmethod
    def calculate_repetition_ratio(text):
        words = text.split()
        if len(words) < 10:
            return 0, {}
        metrics = {}
        word_freq = Counter(words)
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        metrics['word_repetition'] = repeated_words / len(word_freq) if word_freq else 0
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        repeated_bigrams = sum(1 for count in bigram_freq.values() if count > 1)
        metrics['bigram_repetition'] = repeated_bigrams / len(bigram_freq) if bigram_freq else 0
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        trigram_freq = Counter(trigrams)
        repeated_trigrams = sum(1 for count in trigram_freq.values() if count > 1)
        metrics['trigram_repetition'] = repeated_trigrams / len(trigram_freq) if trigram_freq else 0
        overall_repetition = np.mean(list(metrics.values()))
        return overall_repetition, metrics
        
# Запуск ботов
class AdvancedMultiBotPipeline:
    def __init__(self, output_dir="multi_bot_results", max_sentences=None, timeout=300, text_sample_size=50000):
        self.output_dir = output_dir
        self.max_sentences = max_sentences
        self.timeout = timeout
        self.text_sample_size = text_sample_size
        self.metrics_calc = TextMetrics()
        self.net_analyzer = NetworkAnalyzer(lower=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.bots = {
            'markov': {'name': 'Markov Chain (3-gram)', 'complexity': 'Very Low', 'color': '#e74c3c', 'type': 'statistical'},
            'simple_lstm': {'name': 'Simple LSTM', 'complexity': 'Low', 'color': '#f39c12', 'type': 'neural'},
            'better_lstm': {'name': 'Better LSTM (char-level)', 'complexity': 'Medium', 'color': '#3498db', 'type': 'neural'},
            'hybrid': {'name': 'Hybrid N-gram+NN', 'complexity': 'Medium-High', 'color': '#2ecc71', 'type': 'hybrid'},
            'cnn_attention': {'name': 'CNN+Attention', 'complexity': 'Medium-High', 'color': '#9b59b6', 'type': 'neural'},
            'advanced_ngram': {'name': 'Advanced N-gram (GPT-2 alt)', 'complexity': 'High', 'color': '#e67e22', 'type': 'hybrid'},
            'statistical_lm': {'name': 'Statistical LM (BERT alt)', 'complexity': 'High', 'color': '#1abc9c', 'type': 'statistical'}
        }

    @staticmethod
    def get_text_fragment(full_text: str, size: int, method="middle") -> str:
        if len(full_text) <= size:
            return full_text
        if method == "start":
            return full_text[:size]
        elif method == "random":
            start = random.randint(0, len(full_text) - size)
            return full_text[start:start + size]
        else:
            start = (len(full_text) - size) // 2
            return full_text[start:start + size]

    def clean_text_for_bot(self, text: str, lang: str = 'en') -> str:
        text = re.sub(r'\s+', ' ', text)
        if lang == 'rom':
            text = re.sub(r'[^\w\s.,!?;:()\-\'\"ăâîșțĂÂÎȘȚ]', '', text)
        elif lang == 'ru':
            text = re.sub(r'[^\w\s.,!?;:()\-\'\"а-яА-ЯёЁ]', '', text)
        else:
            text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', '', text)
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 10 or line.strip() == '']
        text = '\n'.join(lines)
        return text.strip()

    def _create_simple_bot_text(self, human_text: str, length: int = 3000) -> str:
        words = human_text.split()
        if len(words) > length:
            start = random.randint(0, len(words) - length)
            return ' '.join(words[start:start + length])
        return human_text

    def generate_with_all_bots(self, human_text: str, lang: str = 'en') -> Dict:
        bot_texts = {}
        train_text = self.clean_text_for_bot(human_text, lang)
        train_limit = min(50000, len(train_text))
        train_sample = train_text[:train_limit]
        print(f"Training text: {len(train_sample)} characters (from {len(train_text)} total)\n")

        # 1. Markov
        print("Markov Chain (3-gram)")
        markov = MarkovChainBot(n=3)
        markov.train(train_sample)
        words = train_sample.split()
        start_idx = min(100, len(words)-3)
        seed = ' '.join(words[start_idx:start_idx+3])
        bot_texts['markov'] = markov.generate(seed, length=3000, temperature=0.7)

        # 2. Simple LSTM
        print("Simple LSTM")
        simple_bot = SimpleBotWrapper()
        simple_bot.train(train_sample, epochs=20, hidden_size=128)
        start_text = train_sample[100:200] if len(train_sample) > 200 else train_sample[:100]
        bot_texts['simple_lstm'] = simple_bot.generate(start_text, length=3000, temperature=0.8)

        # 3. Better LSTM
        print("Better LSTM (char-level)")
        better_bot = BetterLSTMBot(hidden_size=256)
        better_bot.train(train_sample, epochs=30)
        start_text = train_sample[100:200] if len(train_sample) > 200 else train_sample[:100]
        bot_texts['better_lstm'] = better_bot.generate(start_text, length=3000, temperature=0.8)

        # 4. Hybrid
        print("Hybrid N-gram + Neural")
        hybrid = HybridBot(n_gram_order=5, neural_dim=128)
        hybrid.train(train_sample, epochs=15, lang=lang)
        words = train_sample.split()
        start_idx = min(100, len(words)-5)
        seed = ' '.join(words[start_idx:start_idx+5])
        bot_texts['hybrid'] = hybrid.generate(seed, length=1500, temperature=0.75)

        # 5. CNN+Attention
        print("CNN + Attention")
        cnn_bot = CNNAttentionBot()
        cnn_bot.train(train_sample, epochs=30)
        words = train_sample.split()
        start_idx = min(100, len(words)-5)
        seed = ' '.join(words[start_idx:start_idx+5])
        bot_texts['cnn_attention'] = cnn_bot.generate(seed, length=1500, temperature=0.8)

        # 6. Advanced N-gram
        print("Advanced N-gram (GPT-2 alternative)")
        adv_ngram = AdvancedNGramBot(max_n=5)
        adv_ngram.train(train_sample, epochs=5, lang=lang)
        words = train_sample.split()
        seed = ' '.join(words[50:55]) if len(words) > 55 else train_sample[:50]
        bot_texts['advanced_ngram'] = adv_ngram.generate(seed, length=1500)

        # 7. Statistical LM
        print("Statistical LM (BERT alternative)")
        stat_lm = StatisticalLMBot()
        stat_lm.train(train_sample, epochs=10)
        words = train_sample.split()
        seed = ' '.join(words[100:110]) if len(words) > 110 else train_sample[:50]
        bot_texts['statistical_lm'] = stat_lm.generate(seed, length=1500, temperature=0.8)
        return bot_texts

    def compare_all_bots(self, human_text: str, bot_texts: Dict, lang: str = 'en') -> Dict:
        all_comparisons = {}
        traj_analyzer = SemanticTrajectoryAnalyzer(word_embeddings=None)
        human_traj = traj_analyzer.get_trajectory(human_text)
        human_traj_metrics = traj_analyzer.compute_trajectory_metrics(human_traj)
        human_network = self.net_analyzer.compute_all(human_text)
        human_words = set(human_text.lower().split())
        human_word_count = len(human_text.split())

    for bot_name, bot_text in bot_texts.items():
        bot_info = self.bots[bot_name]
        bot_words = set(bot_text.lower().split())
        bot_word_count = len(bot_text.split())
        intersection = len(human_words & bot_words)
        union = len(human_words | bot_words)
        jaccard = intersection / union if union > 0 else 0
        metrics_dict = {
            'jaccard_similarity': jaccard,
            'vocabulary_overlap': intersection / len(human_words) if human_words else 0
        }
        try:
            metrics_dict['perplexity'] = self.metrics_calc.calculate_perplexity(bot_text)
        except:
            metrics_dict['perplexity'] = float('inf')
        try:
            metrics_dict['levenshtein_similarity'] = self.metrics_calc.calculate_levenshtein_similarity(human_text, bot_text)
        except:
            metrics_dict['levenshtein_similarity'] = 0
        try:
            complexity, complexity_details = self.metrics_calc.calculate_syntactic_complexity(bot_text)
            metrics_dict['syntactic_complexity'] = complexity
            metrics_dict['complexity_details'] = complexity_details
        except:
            metrics_dict['syntactic_complexity'] = 0
            metrics_dict['complexity_details'] = {}
        try:
            rep_ratio, rep_details = self.metrics_calc.calculate_repetition_ratio(bot_text)
            metrics_dict['repetition_ratio'] = rep_ratio
            metrics_dict['repetition_details'] = rep_details
        except:
            metrics_dict['repetition_ratio'] = 0
            metrics_dict['repetition_details'] = {}
        bot_traj = traj_analyzer.get_trajectory(bot_text)
        bot_traj_metrics = traj_analyzer.compute_trajectory_metrics(bot_traj)
        traj_sim = traj_analyzer.trajectory_similarity(human_traj, bot_traj)
        metrics_dict['trajectory_length'] = bot_traj_metrics['trajectory_length']
        metrics_dict['trajectory_mean_speed'] = bot_traj_metrics['mean_speed']
        metrics_dict['trajectory_tortuosity'] = bot_traj_metrics['tortuosity']
        metrics_dict['trajectory_similarity'] = traj_sim
        human_unique_ratio = len(human_words) / human_word_count if human_word_count > 0 else 0
        bot_unique_ratio = len(bot_words) / bot_word_count if bot_word_count > 0 else 0
        structural_sim = 1 - abs(human_unique_ratio - bot_unique_ratio)
        metrics_dict['structural_similarity'] = structural_sim
        bot_network = self.net_analyzer.compute_all(bot_text)
        metrics_dict['network'] = {'human': human_network, 'bot': bot_network}
        human_metrics = {
            'num_vertices': human_word_count,
            'num_edges': human_word_count * 2,
            'density': 0.01,
            'num_words': human_word_count,
            'unique_words': len(human_words),
            'avg_word_length': np.mean([len(w) for w in human_text.split()]) if human_text.split() else 0
        }
        bot_metrics = {
            'num_vertices': bot_word_count,
            'num_edges': bot_word_count * 2,
            'density': 0.01,
            'num_words': bot_word_count,
            'unique_words': len(bot_words),
            'avg_word_length': np.mean([len(w) for w in bot_text.split()]) if bot_text.split() else 0
        }
        all_comparisons[bot_name] = {
            'name': bot_info['name'],
            'complexity': bot_info['complexity'],
            'type': bot_info['type'],
            'human_metrics': human_metrics,
            'bot_metrics': bot_metrics,
            'comparison': metrics_dict,
            'bot_text': bot_text
        }
        print(f"{bot_info['name']}:")
        print(f"Trajectory Similarity: {traj_sim:.3f}")
        print(f"Repetition: {metrics_dict['repetition_ratio']*100:.1f}%")
        print(f"Network Assortativity: {bot_network['assortativity']:.3f} (human: {human_network['assortativity']:.3f})")
    return all_comparisons

    def visualize_advanced_comparison(self, all_comparisons: Dict, lang: str, human_text: str = ""):
        fig = plt.figure(figsize=(18, 12))
        ax1 = fig.add_subplot(2, 3, 1, projection='polar')
        self._plot_radar_chart(ax1, all_comparisons)
        ax2 = fig.add_subplot(2, 3, 2)
        self._plot_heatmap(ax2, all_comparisons)
        ax3 = fig.add_subplot(2, 3, 3)
        self._plot_trajectory_pca(ax3, all_comparisons, human_text)
        ax4 = fig.add_subplot(2, 3, 4)
        self._plot_similarity_scores(ax4, all_comparisons)
        ax5 = fig.add_subplot(2, 3, 5)
        self._plot_repetition(ax5, all_comparisons)
        ax6 = fig.add_subplot(2, 3, 6)
        self._plot_complexity_quality(ax6, all_comparisons)
        plt.suptitle(f'Multi-Bot Analysis with Semantic Trajectories: {lang.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, f'advanced_comparison_{lang}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_file}")

    def visualize_network_metrics(self, all_comparisons: Dict, lang: str):
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        first_bot = list(all_comparisons.keys())[0]
        human_net = all_comparisons[first_bot]['comparison']['network']['human']
        sorted_bots = sorted(all_comparisons.items(), key=lambda x: x[1]['comparison'].get('composite_score', 0), reverse=True)
        top_bots = sorted_bots[:3]
        ax = axes[0, 0]
        degrees_human = human_net['degree_distribution']
        if degrees_human:
            sorted_deg = np.sort(degrees_human)[::-1]
            ccdf = 1.0 - np.arange(len(sorted_deg)) / len(sorted_deg)
            ax.loglog(sorted_deg, ccdf, 'k-', linewidth=2, label='Human')
        for bot_name, comp_data in top_bots:
            net = comp_data['comparison']['network']['bot']
            degs = net['degree_distribution']
            if degs:
                sorted_deg = np.sort(degs)[::-1]
                ccdf = 1.0 - np.arange(len(sorted_deg)) / len(sorted_deg)
                ax.loglog(sorted_deg, ccdf, color=self.bots[bot_name]['color'], alpha=0.8, label=self.bots[bot_name]['name'][:20])
        ax.set_xlabel('Degree (k)')
        ax.set_ylabel('P(K ≥ k)')
        ax.set_title('Degree Distribution (CCDF)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if human_net['degrees_for_knn']:
            ax.scatter(human_net['degrees_for_knn'], human_net['knn_for_plot'], c='black', alpha=0.4, s=20, label='Human')
        for bot_name, comp_data in top_bots:
            net = comp_data['comparison']['network']['bot']
            if net['degrees_for_knn']:
                ax.scatter(net['degrees_for_knn'], net['knn_for_plot'], color=self.bots[bot_name]['color'], alpha=0.4, s=20, label=self.bots[bot_name]['name'][:20])
        ax.set_xlabel('Node degree k')
        ax.set_ylabel('Mean neighbor degree <k_nn>')
        ax.set_title('Local Assortativity (k_nn(k))')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        max_val = max(max(human_net['degrees_for_knn']+human_net['knn_for_plot']) if human_net['degrees_for_knn'] else 1,
                       max([max(net['degrees_for_knn']+net['knn_for_plot']) for _,cd in top_bots for net in [cd['comparison']['network']['bot']] if net['degrees_for_knn']], default=1))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

        ax = axes[0, 2]
        hub_human = human_net['hubs'][:10]
        words_human = [w for w,_ in hub_human]
        degs_human = [d for _,d in hub_human]
        ax.barh(words_human[::-1], degs_human[::-1], color='gray', alpha=0.7, label='Human')
        best_bot_net = top_bots[0][1]['comparison']['network']['bot']
        hub_bot = best_bot_net['hubs'][:10]
        words_bot = [w for w,_ in hub_bot]
        degs_bot = [d for _,d in hub_bot]
        ax.barh(words_bot[::-1], degs_bot[::-1], color=self.bots[top_bots[0][0]]['color'], alpha=0.5, label=self.bots[top_bots[0][0]]['name'][:20])
        ax.set_xlabel('Degree')
        ax.set_title('Top-10 Hub Words')
        ax.legend(fontsize=7)

        ax = axes[1, 0]
        labels = ['Human']
        values = [human_net['assortativity']]
        colors = ['black']
        for bot_name, comp_data in sorted_bots:
            labels.append(self.bots[bot_name]['name'][:15])
            values.append(comp_data['comparison']['network']['bot']['assortativity'])
            colors.append(self.bots[bot_name]['color'])
        ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_ylabel('Assortativity coefficient')
        ax.set_title('Global Degree Assortativity')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1, 1]
        data_to_plot = [human_net['transition_probs']]
        positions = [1]
        tick_labels = ['Human']
        for i, (bot_name, comp_data) in enumerate(sorted_bots):
            net = comp_data['comparison']['network']['bot']
            if net['transition_probs']:
                data_to_plot.append(net['transition_probs'])
                positions.append(i+2)
                tick_labels.append(self.bots[bot_name]['name'][:10])
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['black'] + [self.bots[bn]['color'] for bn,_ in sorted_bots]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_ylabel('Transition probability P(w_j|w_i)')
        ax.set_title('Edge Weight Distribution')
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1, 2]
        labels = ['Human']
        values = [human_net['mean_degree']]
        colors = ['black']
        for bot_name, comp_data in sorted_bots:
            labels.append(self.bots[bot_name]['name'][:15])
            values.append(comp_data['comparison']['network']['bot']['mean_degree'])
            colors.append(self.bots[bot_name]['color'])
        ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_ylabel('Mean degree')
        ax.set_title('Average Word Co-occurrence Degree')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle(f'Network Analysis: Human vs Bots ({lang.upper()})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, f'network_analysis_{lang}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Network visualization saved to {output_file}")

    def _plot_radar_chart(self, ax, all_comparisons):
        categories = ['Jaccard', 'Levenshtein', 'Complexity', 'Anti-Rep', 'Traj.Sim']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, size=8)
        sorted_bots = sorted(all_comparisons.items(), key=lambda x: x[1]['comparison'].get('composite_score', 0), reverse=True)[:5]
        for bot_name, comp_data in sorted_bots:
            m = comp_data['comparison']
            values = [
                m.get('jaccard_similarity', 0),
                m.get('levenshtein_similarity', 0),
                min(m.get('syntactic_complexity', 0) / 20, 1.0),
                1 - min(m.get('repetition_ratio', 0) * 2, 1.0),
                m.get('trajectory_similarity', 0)
            ]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, color=self.bots[bot_name]['color'], label=self.bots[bot_name]['name'][:20])
            ax.fill(angles, values, alpha=0.1, color=self.bots[bot_name]['color'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=6)
        ax.set_title('Multi-Metric Radar', size=10)

    def _plot_heatmap(self, ax, all_comparisons):
        metrics_names = ['Jaccard', 'Levensh.', 'Complex.', 'Anti-Rep', 'Traj.Sim']
        bot_names = [self.bots[bn]['name'][:15] for bn in all_comparisons.keys()]
        data = []
        for bot_name in all_comparisons.keys():
            m = all_comparisons[bot_name]['comparison']
            row = [
                m.get('jaccard_similarity', 0),
                m.get('levenshtein_similarity', 0),
                min(m.get('syntactic_complexity', 0) / 20, 1.0),
                1 - min(m.get('repetition_ratio', 0) * 2, 1.0),
                m.get('trajectory_similarity', 0)
            ]
            data.append(row)
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(metrics_names)))
        ax.set_xticklabels(metrics_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(bot_names)))
        ax.set_yticklabels(bot_names, fontsize=8)
        for i in range(len(bot_names)):
            for j in range(len(metrics_names)):
                ax.text(j, i, f'{data[i][j]:.2f}', ha="center", va="center", color="black", fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title('Metrics Heatmap', fontsize=10)

    def _plot_trajectory_pca(self, ax, all_comparisons, human_text):
        traj_analyzer = SemanticTrajectoryAnalyzer()
        human_traj = traj_analyzer.get_trajectory(human_text)
        if len(human_traj) == 0:
            ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center')
            return
        all_emb = [human_traj]
        labels = ['Human']
        colors = ['black']
        for bot_name, comp_data in all_comparisons.items():
            bot_text = comp_data.get('bot_text', '')
            bot_traj = traj_analyzer.get_trajectory(bot_text)
            if len(bot_traj) > 0:
                all_emb.append(bot_traj)
                labels.append(self.bots[bot_name]['name'][:15])
                colors.append(self.bots[bot_name]['color'])
        if len(all_emb) < 2:
            ax.text(0.5, 0.5, 'Not enough trajectories', ha='center', va='center')
            return
        stacked = np.vstack(all_emb)
        pca = PCA(n_components=2)
        pca.fit(stacked)
        for i, traj in enumerate(all_emb):
            proj = pca.transform(traj)
            ax.plot(proj[:, 0], proj[:, 1], 'o-', color=colors[i], alpha=0.7, markersize=4, label=labels[i])
            ax.scatter(proj[0, 0], proj[0, 1], color=colors[i], s=50, marker='>')
            ax.scatter(proj[-1, 0], proj[-1, 1], color=colors[i], s=50, marker='s')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_title('Semantic Trajectories (PCA)')
        ax.grid(alpha=0.3)

    def _plot_similarity_scores(self, ax, all_comparisons):
        bot_names = list(all_comparisons.keys())
        similarities = [comp['comparison'].get('structural_similarity', 0) * 100 for comp in all_comparisons.values()]
        colors = [self.bots[bn]['color'] for bn in bot_names]
        labels = [self.bots[bn]['name'][:25] for bn in bot_names]
        ax.barh(range(len(bot_names)), similarities, color=colors, alpha=0.7)
        ax.set_yticks(range(len(bot_names)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Similarity to Human (%)')
        ax.set_title('Text Similarity Scores')
        for i, v in enumerate(similarities):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)

    def _plot_repetition(self, ax, all_comparisons):
        bots_data = []
        for bot_name, data in all_comparisons.items():
            rep_ratio = data['comparison'].get('repetition_ratio', 0)
            bots_data.append((bot_name, rep_ratio))
        names = [self.bots[bn]['name'][:15] for bn, _ in bots_data]
        values = [v * 100 for _, v in bots_data]
        colors = [self.bots[bn]['color'] for bn, _ in bots_data]
        ax.barh(names, values, color=colors, alpha=0.7)
        ax.set_xlabel('Repetition Rate (%)')
        ax.set_title('Text Repetition Analysis\n(lower is better)')
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='Warning')
        ax.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='High')
        ax.legend(fontsize=8)

    def run_advanced_analysis(self, human_text: str, lang: str = 'en') -> Dict:
        start_time = time.time()
        bot_texts = self.generate_with_all_bots(human_text, lang)
        all_comparisons = self.compare_all_bots(human_text, bot_texts, lang)
        self.visualize_advanced_comparison(all_comparisons, lang, human_text)
        self.visualize_network_metrics(all_comparisons, lang)
        self.save_results(all_comparisons, lang)
        elapsed = time.time() - start_time
        return all_comparisons

    def save_results(self, all_comparisons: Dict, lang: str):
        for bot_name, comp_data in all_comparisons.items():
            bot_text = comp_data['bot_text']
            text_file = os.path.join(self.output_dir, f'{bot_name}_text_{lang}.txt')
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(bot_text)
        metrics_data = {}
        for bot_name, comp_data in all_comparisons.items():
            comp = comp_data['comparison']
            net = comp['network']
            metrics_data[bot_name] = {
                'name': comp_data['name'],
                'complexity': comp_data['complexity'],
                'type': comp_data['type'],
                # composite_score удалён
                'jaccard_similarity': float(comp.get('jaccard_similarity', 0)),
                'levenshtein_similarity': float(comp.get('levenshtein_similarity', 0)),
                'perplexity': float(comp.get('perplexity', 0)),
                'syntactic_complexity': float(comp.get('syntactic_complexity', 0)),
                'repetition_ratio': float(comp.get('repetition_ratio', 0)),
                'structural_similarity': float(comp.get('structural_similarity', 0)),
                'trajectory_similarity': float(comp.get('trajectory_similarity', 0)),
                'trajectory_length': float(comp.get('trajectory_length', 0)),
                'trajectory_tortuosity': float(comp.get('trajectory_tortuosity', 0)),
                'network_assortativity': net['bot']['assortativity'],
                'network_mean_degree': net['bot']['mean_degree'],
                'network_mean_trans_prob': net['bot']['mean_trans_prob'],
                'network_top_hubs': net['bot']['hubs'][:10]
            }
        metrics_file = os.path.join(self.output_dir, f'advanced_metrics_{lang}.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        print(f"✅ Results saved to {self.output_dir}")

def collect_pdfs_from_folder(folder_path: str) -> List[str]:
    if not os.path.isdir(folder_path):
        return []
    return sorted(glob.glob(os.path.join(folder_path, "*.pdf")))

def extract_text_from_pdfs(pdf_paths: List[str], lang: str, pdf_to_text_func) -> Tuple[str, List[str]]:
    all_text = []
    failed_files = []
    lang_codes = [lang]
    if lang == 'ru':
        lang_codes.append('rus')
    elif lang == 'rom':
        lang_codes.append('ro')
    for pdf_path in pdf_paths:
        file_text = ""
        success = False
        for code in lang_codes:
            if code in ('rom', 'ro'):
                text = pdf_to_text_func(pdf_path, lang=code, start_page=34)
            else:
                text = pdf_to_text_func(pdf_path, lang=code)
            if text and len(text.strip()) > 100:
                file_text = text
                success = True
                break
        if success:
            all_text.append(file_text)
        else:
            failed_files.append(os.path.basename(pdf_path))
    full_text = "\n".join(all_text)
    return full_text, failed_files

def run_literature_analysis(pipeline: AdvancedMultiBotPipeline, lang: str, pdf_to_text_func):
    folder_name = f"{lang}_science"
    if not os.path.isdir(folder_name):
        print(f"Folder '{folder_name}' not found.")
        return None
    pdf_files = collect_pdfs_from_folder(folder_name)
    if not pdf_files:
        print(f"No PDF files in '{folder_name}'.")
        return None
    full_text, failed_files = extract_text_from_pdfs(pdf_files, lang, pdf_to_text_func)
    # Берем фрагмент размером text_sample_size (50000 символов)
    full_text = pipeline.get_text_fragment(full_text, pipeline.text_sample_size)
    print(f"Total text for analysis: {len(full_text)} characters")
    os.makedirs(pipeline.output_dir, exist_ok=True)
    lit_output_dir = os.path.join(pipeline.output_dir, f"literature_{lang}")
    os.makedirs(lit_output_dir, exist_ok=True)
    os.makedirs(lit_output_dir, exist_ok=True)
    original_output_dir = pipeline.output_dir
    pipeline.output_dir = lit_output_dir
    results = pipeline.run_advanced_analysis(full_text, lang=lang)
    pipeline.output_dir = original_output_dir
    return results

def compare_bible_vs_literature(pipeline: AdvancedMultiBotPipeline, lang: str, bible_results: Dict, lit_results: Optional[Dict]):
    if lit_results is None:
        return
    os.makedirs(pipeline.output_dir, exist_ok=True)
    common_bots = set(bible_results.keys()) & set(lit_results.keys())
    comp_data = []
    for bot_name in sorted(common_bots, key=lambda b: pipeline.bots[b]['name']):
        b_score = bible_results[bot_name]['comparison'].get('composite_score', 0) * 100
        l_score = lit_results[bot_name]['comparison'].get('composite_score', 0) * 100
        b_traj = bible_results[bot_name]['comparison'].get('trajectory_similarity', 0)
        l_traj = lit_results[bot_name]['comparison'].get('trajectory_similarity', 0)
        b_repet = bible_results[bot_name]['comparison'].get('repetition_ratio', 0) * 100
        l_repet = lit_results[bot_name]['comparison'].get('repetition_ratio', 0) * 100
        comp_data.append({
            'bot': pipeline.bots[bot_name]['name'],
            'bible_score': b_score,
            'lit_score': l_score,
            'bible_traj': b_traj,
            'lit_traj': l_traj,
            'bible_repet': b_repet,
            'lit_repet': l_repet
        })
    print(f"\n{'Bot':<30} {'Bible Score':>12} {'Lit Score':>12} {'Diff':>8}")
    for d in comp_data:
        diff = d['lit_score'] - d['bible_score']
        print(f"{d['bot']:<30} {d['bible_score']:>11.1f}% {d['lit_score']:>11.1f}% {diff:>+7.1f}%")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    bots_names = [d['bot'] for d in comp_data]
    bible_scores = [d['bible_score'] for d in comp_data]
    lit_scores = [d['lit_score'] for d in comp_data]
    x = np.arange(len(bots_names))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, bible_scores, width, label='Bible', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, lit_scores, width, label='Literature', color='#e67e22', alpha=0.8)
    ax.set_ylabel('Composite Score (%)')
    ax.set_title(f'Overall Quality: Bible vs Literature ({lang.upper()})')
    ax.set_xticks(x)
    ax.set_xticklabels(bots_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax = axes[1]
    traj_b = [d['bible_traj'] for d in comp_data]
    traj_l = [d['lit_traj'] for d in comp_data]
    ax.bar(x - width/2, traj_b, width, label='Bible', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, traj_l, width, label='Literature', color='#e67e22', alpha=0.8)
    ax.set_ylabel('Trajectory Similarity')
    ax.set_title('Semantic Trajectory Similarity')
    ax.set_xticks(x)
    ax.set_xticklabels(bots_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax = axes[2]
    rep_b = [d['bible_repet'] for d in comp_data]
    rep_l = [d['lit_repet'] for d in comp_data]
    ax.bar(x - width/2, rep_b, width, label='Bible', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, rep_l, width, label='Literature', color='#e67e22', alpha=0.8)
    ax.set_ylabel('Repetition Rate (%)')
    ax.set_title('Text Repetition (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(bots_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'Bible vs Literature – {lang.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(pipeline.output_dir, f'bible_vs_literature_{lang}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    json_path = os.path.join(pipeline.output_dir, f'bible_vs_literature_{lang}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comp_data, f, indent=2)

def run_advanced_analysis_all_languages():
    os.makedirs("multi_bot_results", exist_ok=True)
    pipeline = AdvancedMultiBotPipeline(
        output_dir="multi_bot_results",
        max_sentences=None,
        timeout=300,
        text_sample_size=50000
    )
    bible_files = {
        "en": "Bible_eng.pdf",
        "ru": "Bible_rus.pdf",
        "rom": "Bible_rom.pdf"
    }
    available_bibles = {}
    for lang, filename in bible_files.items():
        if os.path.exists(filename):
            available_bibles[lang] = filename
        else:
            print(f"Not found: {filename}")
    if not available_bibles:
        print("No Bibles found")
        return

    bible_results_all = {}
    for lang, filename in available_bibles.items():
      if lang == "rom":
          full_text = pdf_to_text(filename, lang=lang, start_page=34)
      else:
          full_text = pdf_to_text(filename, lang=lang)
      text = pipeline.get_text_fragment(full_text, pipeline.text_sample_size)
      results = pipeline.run_advanced_analysis(text, lang=lang)
      bible_results_all[lang] = results
      rankings = sorted(results.items(),
                        key=lambda x: x[1]['comparison'].get('composite_score', 0),
                        reverse=True)
      for rank, (bot_name, comp_data) in enumerate(rankings, 1):
          bot_info = pipeline.bots[bot_name]
          comp = comp_data['comparison']
          composite_score = comp.get('composite_score', 0)
          print(f"{bot_info['name']}")
          print(f"Composite Score: {composite_score*100:.1f}%")
          print(f"Trajectory Similarity: {comp.get('trajectory_similarity', 0):.3f}")
          print(f"Repetition: {comp.get('repetition_ratio', 0)*100:.1f}%")
    
    # Анализ научной литературы
    for lang in ["en", "ru"]:
        lit_results = run_literature_analysis(pipeline, lang, pdf_to_text)
        if lit_results:
            lang_key = 'en' if lang == 'en' else 'ru'
            if lang_key in bible_results_all:
                compare_bible_vs_literature(pipeline, lang, bible_results_all[lang_key], lit_results)
    print(f"Результаты сохранены в {pipeline.output_dir}")

if __name__ == "__main__":
    run_advanced_analysis_all_languages()
