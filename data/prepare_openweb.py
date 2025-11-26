import os, pickle
import numpy as np
from pathlib import Path
from datasets import load_dataset
import tiktoken

# DooHickeys
MAX_DOCS = 20000
VAL_FRACTION = 0.01
MIN_VAL_TOKENS = 512_000
OUT_DIR = Path(os.path.join(os.path.dirname(__file__),"openweb"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

def doc_iter(max_docs):
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= max_docs: break
        txt = ex.get("text")
        if txt: 
            yield txt

if __name__ == "__main__":
    total_tokens, num_docs = 0, 0
    for txt in doc_iter(MAX_DOCS):
        total_tokens += len(enc.encode_ordinary(txt))+1
        num_docs += 1

    if total_tokens == 0:
        raise SystemExit("No tokens. Check dataset/network.")

    val_tokens = min(
    max(int(total_tokens * VAL_FRACTION), MIN_VAL_TOKENS),
    max(1, total_tokens // 10)
)

    train_tokens = total_tokens - val_tokens

    train_mm = np.memmap(OUT_DIR / "train.bin", dtype=np.uint16, mode="w+", shape=(train_tokens,))
    val_mm   = np.memmap(OUT_DIR / "val.bin",   dtype=np.uint16, mode="w+", shape=(val_tokens,))

    t_idx = v_idx = seen = 0
    boundary = train_tokens

    def write_ids(ids):
        global t_idx , v_idx , seen
        arr = np.asarray(ids, dtype=np.uint16)
        start = 0
        while start < len(arr):
            remain_train = boundary - seen
            if remain_train > 0:
                take = min(remain_train, len(arr) - start)
                train_mm[t_idx:t_idx+take] = arr[start:start+take]
                t_idx += take; seen += take; start += take
            else:
                take = len(arr) - start
                if take == 0: 
                    break
                val_mm[v_idx:v_idx+take] = arr[start:start+take]
                v_idx += take; seen+= take; start+= take

    buf, buf_tok = [], 0
    BUF_TARGET = 256_000


    for txt in doc_iter(MAX_DOCS):
        ids = enc.encode_ordinary(txt)
        ids.append(enc.eot_token)
        buf.append(ids); buf_tok += len(ids)

        if buf_tok >= BUF_TARGET:
            flat = [x for seq in buf for x in seq]
            write_ids(flat)
            buf, buf_tok = [], 0

    if buf_tok:
        flat = [x for seq in buf for x in seq]
        write_ids(flat)

    train_mm.flush(); val_mm.flush()

    with open(OUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump({
            "dataset": "openwebtext",
            "encoding": "gpt2",
            "vocab_size": enc.n_vocab,
            "eot_token": enc.eot_token,
            "num_docs": num_docs,
            "total_tokens": int(total_tokens),
            "train_tokens": int(train_tokens),
            "val_tokens": int(val_tokens),
            "max_docs": MAX_DOCS,
            "streaming": True,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done. train:{train_tokens:,} val:{val_tokens:,} docs:{num_docs:,} → {OUT_DIR} ")

