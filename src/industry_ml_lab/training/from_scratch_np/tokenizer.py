"""BPE (Byte Pair Encoding) tokenizer implementation.

This module implements a BPE tokenizer from scratch, following the original
GPT-2 tokenizer approach. It learns merge rules from training data and
can encode/decode text to/from token IDs.

No deep learning frameworks are used - pure NumPy only.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Tuple, Optional
import json
import re


class BPETokenizer:
    """Byte Pair Encoding tokenizer for language models.
    
    BPE is a subword tokenization algorithm that:
    1. Starts with individual bytes/characters
    2. Iteratively merges the most frequent pairs
    3. Creates a vocabulary of subword units
    
    This implementation follows the GPT-2 tokenizer approach with:
    - Byte-level encoding (handles all Unicode)
    - Regex-based pre-tokenization
    - Learned merge rules
    
    Args:
        vocab_size: Target vocabulary size (default: 50257 for GPT-2)
        merge_rules: Optional pre-trained merge rules
    """
    
    _GPT2_PATTERN = re.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'(?i:[sdmt]|ll|ve|re)|"""
        r"""[^\r\n\p{L}\p{N}]?[\p{L}]+|[\p{A}]+|[\p{N}]|(?=[\s.,;?!])|(?![\s.,;?!])|"""
        r"""[\s.,;?!]|\s+(?!\S)|\s+"""
    )
    
    def __init__(
        self,
        vocab_size: int = 50257,
        merge_rules: Optional[Dict[Tuple[int, int], int]] = None,
        vocab: Optional[Dict[int, bytes]] = None
    ):
        self.vocab_size = vocab_size
        
        self.merge_rules: Dict[Tuple[int, int], int] = merge_rules or {}
        self.reverse_rules: Dict[int, Tuple[int, int]] = {}
        for (left, right), new_id in self.merge_rules.items():
            self.reverse_rules[new_id] = (left, right)
        
        self.vocab: Dict[int, bytes] = vocab or self._build_base_vocab()
        self.reverse_vocab: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        
        self.bos_token: int = self.vocab_size - 2 if vocab is None else self._get_token_id("<|startoftext|>")
        self.eos_token: int = self.vocab_size - 1 if vocab is None else self._get_token_id("<|endoftext|>")
    
    def _build_base_vocab(self) -> Dict[int, bytes]:
        """Build base vocabulary with all bytes + special tokens."""
        vocab = {i: bytes([i]) for i in range(256)}
        vocab[256] = b"<|startoftext|>"
        vocab[257] = b"<|endoftext|>"
        return vocab
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID by string representation."""
        token_bytes = token.encode("utf-8")
        return self.reverse_vocab.get(token_bytes, -1)
    
    def train(self, text: str, verbose: bool = False) -> None:
        """Train BPE merge rules on text corpus.
        
        Args:
            text: Training text corpus
            verbose: Print progress information
        """
        if verbose:
            print(f"Training BPE tokenizer on {len(text):,} characters...")
        
        words = self._pre_tokenize(text)
        if verbose:
            print(f"Found {len(words):,} unique words")
        
        freq = self._get_word_frequencies(words)
        vocab = self._init_vocab_from_freq(freq)
        merge_rules = self._learn_merge_rules(freq, vocab, verbose)
        
        self.merge_rules = merge_rules
        self.reverse_rules = {new_id: (left, right) for (left, right), new_id in merge_rules.items()}
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        
        if verbose:
            print(f"Final vocabulary size: {len(vocab)}")
            print(f"Merge rules learned: {len(merge_rules)}")
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into words using GPT-2 regex."""
        return self._GPT2_PATTERN.findall(text)
    
    def _get_word_frequencies(self, words: List[str]) -> Dict[str, int]:
        """Count frequency of each word."""
        return collections.Counter(words)
    
    def _init_vocab_from_freq(self, freq: Dict[str, int]) -> Dict[int, bytes]:
        """Initialize vocabulary from word frequencies."""
        vocab = {i: bytes([i]) for i in range(256)}
        vocab[256] = b"<|startoftext|>"
        vocab[257] = b"<|endoftext|>"
        
        for word in freq:
            for byte in word.encode("utf-8"):
                if byte not in vocab:
                    vocab[byte] = bytes([byte])
        
        return vocab
    
    def _learn_merge_rules(
        self,
        freq: Dict[str, int],
        vocab: Dict[int, bytes],
        verbose: bool = False
    ) -> Dict[Tuple[int, int], int]:
        """Learn BPE merge rules from word frequencies."""
        def get_pairs(word: List[int]) -> Dict[Tuple[int, int], int]:
            pairs = collections.defaultdict(int)
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
            return pairs
        
        word_tokens = {}
        for word in freq:
            word_tokens[word] = [vocab[b] for b in word.encode("utf-8")]
        
        merge_id = 258
        merge_rules = {}
        
        while len(vocab) < self.vocab_size:
            pair_freq = collections.defaultdict(int)
            for word, tokens in word_tokens.items():
                pairs = get_pairs(tokens)
                for pair, count in pairs.items():
                    pair_freq[pair] += count
            
            if not pair_freq:
                break
            
            best_pair = max(pair_freq.items(), key=lambda x: x[1])
            pair = best_pair[0]
            
            if pair in self.reverse_vocab:
                continue
            
            new_token = vocab[pair[0]] + vocab[pair[1]]
            vocab[merge_id] = new_token
            self.reverse_vocab[new_token] = merge_id
            
            merge_rules[pair] = merge_id
            self.reverse_rules[merge_id] = pair
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merge_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            word_tokens[word] = new_tokens
            merge_id += 1
            
            if verbose and merge_id % 1000 == 0:
                print(f"  Merge {merge_id}: {pair} -> {merge_id} (freq: {best_pair[1]})")
        
        return merge_rules
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        words = self._pre_tokenize(text)
        
        tokens = []
        for word in words:
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
        
        if add_bos:
            tokens.insert(0, self.bos_token)
        if add_eos:
            tokens.append(self.eos_token)
        
        return tokens
    
    def _encode_word(self, word: str) -> List[int]:
        """Encode a single word to token IDs."""
        word_bytes = word.encode("utf-8")
        tokens = [b for b in word_bytes]
        
        while len(tokens) >= 2:
            best_merge = None
            best_score = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_rules:
                    score = self.merge_rules[pair]
                    if score > best_score:
                        best_score = score
                        best_merge = i
            
            if best_merge is None:
                break
            
            new_id = self.merge_rules[(tokens[best_merge], tokens[best_merge + 1])]
            tokens = tokens[:best_merge] + [new_id] + tokens[best_merge + 2:]
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        text_bytes = b""
        for token in tokens:
            if token in self.vocab:
                text_bytes += self.vocab[token]
        
        try:
            return text_bytes.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return text_bytes.decode("latin-1", errors="replace")
    
    def save(self, path: str) -> None:
        """Save tokenizer to JSON file."""
        state = {
            "vocab_size": self.vocab_size,
            "merge_rules": {f"{k[0]}_{k[1]}": v for k, v in self.merge_rules.items()},
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        merge_rules = {}
        for key, value in state["merge_rules"].items():
            left, right = map(int, key.split("_"))
            merge_rules[(left, right)] = value
        
        vocab = {}
        for key, value in state["vocab"].items():
            vocab[int(key)] = bytes.fromhex(value)
        
        tokenizer = cls(
            vocab_size=state["vocab_size"],
            merge_rules=merge_rules,
            vocab=vocab
        )
        tokenizer.bos_token = state["bos_token"]
        tokenizer.eos_token = state["eos_token"]
        
        return tokenizer
    
    @property
    def vocab_size_actual(self) -> int:
        """Actual vocabulary size (may differ from target)."""
        return len(self.vocab)
    
    def __len__(self) -> int:
        return self.vocab_size_actual
