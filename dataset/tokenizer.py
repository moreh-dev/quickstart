from tokenizers.implementations import SentencePieceBPETokenizer


class Tokenizer:

    def __init__(self, vocab_file_path, merge_file_path, extra_ids):
        self.tokenizer = SentencePieceBPETokenizer(vocab_file_path, merge_file_path)
        self.extra_id_start = self.tokenizer.get_vocab_size()
        self.extra_ids = extra_ids

        sentinels = [f'<mask_{i}>' for i in range(self.extra_ids)]
        self.tokenizer.add_tokens(sentinels)

        self.unknown_token = self.tokenizer.token_to_id("<unk>")
        self._pad_token = "<pad>"
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.init_kwargs = {}
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}
        self.unexpected_sep_token = ['<pad>', '<unk>', '<eos>', '<sos>']

        self.encoder = self.tokenizer.get_vocab()
        self.decoder = dict(map(reversed, self.encoder.items()))

    def tokenize(self, text):
        if text in self.unexpected_sep_token:
            return text
        return self.tokenizer.encode(text).tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        if isinstance(tokens, str):
            if tokens in self.encoder:
                return self.encoder[tokens]
            else:
                return self.unknown_token
        for token in tokens:
            if token in self.encoder:
                ids.append(self.encoder[token])
            else:
                ids.append(self.unknown_token)
        return ids

    def convert_ids_to_tokens(self, ids):
        sentence = ''
        for id_ in ids:
            sentence += self.decoder[id_]
        sentence = sentence.replace('▁', ' ')
        return sentence.strip()

    def build_inputs_with_special_tokens(self, ids):
        return ids

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def add_special_tokens(self, new_tokens):
        self.tokenizer.add_special_tokens(new_tokens)
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = dict(map(reversed, self.encoder.items()))

    def add_tokens(self, new_tokens):
        self.tokenizer.add_tokens(new_tokens)
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = dict(map(reversed, self.encoder.items()))
