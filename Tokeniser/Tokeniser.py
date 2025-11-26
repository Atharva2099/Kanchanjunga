
import tiktoken
import pickle

class Tokeniser:
    def __init__(self,encoding_name="gpt2"):
        #loading tokeniser
        self.enc = tiktoken.get_encoding(encoding_name)

        #self vocab and end of sequence token
        self.n_vocab = self.enc.n_vocab
        self.eot_token = self.enc.eot_token

    def encode(self,text,allowed_special=None):
        #if empty assume False , else whatever is passed (default is false)
        if allowed_special is None:
            return self.enc.encode_ordinary(text)
        else:
            return self.enc.encode_ordinary(text, allowed_special=allowed_special)

    def decode(self,ids):
        return self.enc.decode(ids)

    

if __name__ =="__main__":
    tokeniser = Tokeniser()
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert tokeniser.n_vocab == meta["vocab_size"], "Tokenizer vocab mismatch."
    text = "Hello Kanchanjunga!!!"
    encoded = tokeniser.encode(text)
    decoded = tokeniser.decode(encoded)

    print(f"Vocab size:{tokeniser.n_vocab}")
    print(f"Input: {text}")
    print(f"Encoded text: {encoded}")
    print(f"Decoded: {decoded}")

        