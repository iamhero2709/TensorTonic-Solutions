import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        # Step 1: add special tokens first
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        for token in special_tokens:
            self.word_to_id[token] = self.vocab_size
            self.id_to_word[self.vocab_size] = token
            self.vocab_size += 1
        
        # Step 2: collect unique words
        unique_words = set()
        for text in texts:
            words = text.split()
            unique_words.update(words)
        
        # Step 3: assign IDs
        for word in unique_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        words = text.split()
        ids = []
        
        for word in words:
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.word_to_id[self.unk_token])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        words = []
        
        for i in ids:
            if i in self.id_to_word:
                words.append(self.id_to_word[i])
            else:
                words.append(self.unk_token)
        
        return " ".join(words)