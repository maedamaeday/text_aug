from transformers import BertJapaneseTokenizer, BertForMaskedLM, BertModel
import numpy as np
import torch
import argparse

from typing import List, Optional

def_n_generate = 10
def_n_max_try = 100
def_mask_frac = 0.15
def_cos_thre = 0.9
def_n_cand = 10
def_seed = 0
def_batch_size = 8

class SentenceBertJapanese:
    def __init__(
            self,
            model_name_or_path:str,
            device:Optional[str] = None,
    ):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            model_name_or_path,
        )
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(
            self,
            model_output:str,
            attention_mask:torch.Tensor,
    ):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return (
            torch.sum(token_embeddings * input_mask_expanded, 1)
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

    @torch.no_grad()
    def encode(
            self,
            sentences:List[str],
            batch_size:int = def_batch_size,
            as_numpy:bool = False, 
    ):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch,
                padding="longest", 
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output,
                encoded_input["attention_mask"],
            ).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return (torch.stack(all_embeddings).numpy()
                if as_numpy else torch.stack(all_embeddings))

    
def main(
        input_sentence:str,
        n_generate:int = def_n_generate,
        n_max_try:int = def_n_max_try,
        mask_frac:float = def_mask_frac,
        n_cand:int = def_n_cand,
        cos_thre:float = def_cos_thre,
        seed:int = def_seed,
        verbose:bool = False,
) -> List[str]:

    base_model_path = "cl-tohoku/bert-base-japanese"
    sbert_model_path = "sonoisa/sentence-bert-base-ja-mean-tokens"

    device = "cpu"
    
    tokenizer = BertJapaneseTokenizer.from_pretrained(base_model_path)
    masked_lang_model = BertForMaskedLM.from_pretrained(
        base_model_path
    ).to(device)
    #sbert_model = BertModel.from_pretrained(sbert_model_path)
    sbert_model = SentenceBertJapanese(sbert_model_path)

    input_tokenized = tokenizer.tokenize(input_sentence)
    if verbose:
        print(f"{input_tokenized}")
        print("----------")
    n_token = len(input_tokenized)
    not_yet_masked = (
        ["[CLS]"]+input_tokenized+["[SEP]"]
        +input_tokenized+["[SEP]"]
    )
    print(not_yet_masked)
    
    input_encoded = sbert_model.encode([input_sentence],as_numpy=True)[0]
    print(input_encoded.shape)
    input_norm = np.sqrt(np.sum(input_encoded*input_encoded))
    print(input_norm)
    
    rng = np.random.default_rng(seed=seed) 
    
    generated_list = []
    segment_ids = [0]*(2+n_token)+[1]*(1+n_token)
    segments_tensor = torch.tensor([segment_ids]).to(device)
    for i_try in range(n_max_try):
        if verbose:
            print(f"i_try = {i_try} : n_generated = {len(generated_list)}")
        while True:
            n_mask = rng.poisson( mask_frac*n_token )
            if n_mask>0:
                break
        mask_indices = np.random.choice(
            n_token,
            n_mask,
            replace=False,
        )
        if verbose:
            print(f" mask_indices = {mask_indices}")
        generated = [token for token in not_yet_masked]
        print(generated)
        for mask_index in mask_indices:
            masked_token = input_tokenized[mask_index]
            i_masked = n_token+2+mask_index
            generated[i_masked] = "[MASK]"
            token_ids = tokenizer.convert_tokens_to_ids(generated)
            tokens_tensor = torch.tensor([token_ids]).to(device)
            print(generated)
            with torch.no_grad():
                prediction = masked_lang_model(
                    tokens_tensor,
                    token_type_ids=segments_tensor,
                )[0]
            topk_score, topk_index = torch.topk(
                prediction[0, i_masked],
                n_cand,
            )
            topk_index = topk_index.tolist()
            while True:
                i_cand = rng.choice(topk_index)
                cand_token = tokenizer.convert_ids_to_tokens([i_cand])[0]
                if cand_token!=masked_token:
                    generated[i_masked] = cand_token
                    break
            print(generated)

        print(["".join(generated)])
        generated_encoded = sbert_model.encode(
            ["".join(generated)],
            as_numpy=True,
        )[0]
        generated_norm = np.sqrt(np.sum(generated_encoded*generated_encoded))
        cos = np.sum(generated_encoded*input_encoded)/generated_norm/input_norm
        print(generated_norm, cos, cos_thre)
        if cos>cos_thre:
            generated_list.append(generated)
            if len(generated_list)>=n_generate:
                if verbose:
                    print(f"{i_try} : {generated}")
                break

    return generated_list
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Japanese text generation with similar meaning "
            "to the input sentence using masked language model"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_sentence",
        nargs="+",
        type=str,
        help="input sentence",
    )
    parser.add_argument(
        "-n", "--n_generate",
        type=int,
        default=def_n_generate,
        help="the number of sentences to be generated",
    )
    parser.add_argument(
        "-t", "--n_max_try",
        type=int,
        default=def_n_max_try,
        help="the number of maximum trials",
    )
    parser.add_argument(
        "-f", "--mask_frac",
        type=int,
        default=def_mask_frac,
        help="fraction of the number of masked words",
    )
    parser.add_argument(
        "-c", "--cos_thre",
        type=float,
        default=def_cos_thre,
        help=(
            "threshold of cosine similarity to qualify the same meaning "
            "of generated sentences with the input one"
        ),
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="random number seed for numpy random number generator",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="show messages during processes",
    )

    args = parser.parse_args()
    main(
        input_sentence=" ".join(args.input_sentence),
        n_generate=args.n_generate,
        n_max_try=args.n_max_try,
        mask_frac=args.mask_frac,
        cos_thre=args.cos_thre,
        seed=args.seed,
        verbose=args.verbose,
    )
