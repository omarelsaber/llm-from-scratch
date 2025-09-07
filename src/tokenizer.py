# src/tokenizer.py
"""
Simple wrapper around OpenAI's tiktoken BPE for tokenization encode/decode.
"""
try:
    import tiktoken
except Exception as e:
    raise RuntimeError("tiktoken not installed. Run: pip install tiktoken") from e

class BPEFastTokenizer:
    def __init__(self, encoding_name: str = "gpt2", allowed_special=None):
        allowed_special = allowed_special or {"<|endoftext|>"}
        self.enc = tiktoken.get_encoding(encoding_name)
        self.allowed_special = allowed_special
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str):
        # returns list[int]
        return self.enc.encode(text, allowed_special=self.allowed_special)

    def decode(self, ids):
        # accepts list[int] or bytes-like
        return self.enc.decode(ids)

if __name__ == "__main__":
    tk = BPEFastTokenizer()
    txt = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    ids = tk.encode(txt)
    print("len ids:", len(ids))
    print("decoded:", tk.decode(ids))
