import os
import sys
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import mlxu

from ttt.models.model import ModelConfig, CausalLM
from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.infra.jax_utils import (
    next_rng, 
    get_float_dtype_by_name, 
    set_random_seed, 
    make_shard_and_gather_fns, 
    match_partition_rules
)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize
import argparse

def cosine_similarity(a, b):
    dot_product = jnp.dot(a, b.T)
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b, axis=-1)
    return dot_product / (norm_a * norm_b)

def draw_bar(val, length=20):
    filled = int(val * length)
    return "[" + "=" * filled + " " * (length - filled) + "]"

def run_inference(model_path, test_cases, ttt_lr_mult=0.0, max_new_tokens=20):
    print("=" * 80)
    print(f"""
    MODEL: {model_path}
    TTT LR MULT: {ttt_lr_mult}
    """)
    print("=" * 80)

    metadata = mlxu.load_pickle(os.path.join(model_path, "metadata.pkl"))
    model_config = ModelConfig(**metadata['model_config'])
    
    SEQ_LEN = 1024
    model_config.max_sequence_length = SEQ_LEN
    model_config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    BOS = tokenizer.bos_token_id or 128000
    
    mesh = model_config.get_jax_mesh("1,1,1")
    with mesh:
        dtype = get_float_dtype_by_name('bf16')
        model = CausalLM(model_config, dtype=dtype)
        
        def init_fn(rng):
            return model.init(rng, jnp.zeros((1, SEQ_LEN), dtype=jnp.int32), 
                             attention_mask=jnp.ones((1, SEQ_LEN), dtype=jnp.int32))
                             
        variables_shape = jax.eval_shape(init_fn, next_rng())
        partition_specs = match_partition_rules(model_config.get_partition_rules(), variables_shape)
        sharder = make_shard_and_gather_fns(partition_specs, variables_shape)
        
        print(f"Loading checkpoint...")
        params = StreamingCheckpointer.load_checkpoint(
            os.path.join(model_path, "streaming_train_state"), 
            target=variables_shape['params'], 
            shard_fns=sharder[0]['params'], remove_dict_prefix=('params', 'params')
        )
        embeddings = params['model']['wte']['embedding']
        
        print(f"Model Ready. Starting {len(test_cases)} cases.\n")

        for case in test_cases:
            print(f"Domain: {case['domain']}")
            
            context = case['context']
            prompt = case['prompt']
            target_str = case['target']

            sentences = context.split('.')
            tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sentences)]

            vec_size = 50
            d2v_model = Doc2Vec(vector_size=vec_size, window=2, min_count=1, workers=4, epochs=40)
            d2v_model.build_vocab(tagged_data)
            d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
            
            prompt_sentences = prompt.strip().split('.')
            last_sentence = prompt_sentences[-1] if prompt_sentences[-1] else prompt_sentences[-2]
            prompt_vec = d2v_model.infer_vector(word_tokenize(last_sentence.lower()))

            cids = tokenizer.encode(context, add_special_tokens=False)
            pids = tokenizer.encode(prompt, add_special_tokens=False)
            
            full_ids = [BOS] + cids + pids
            current_ids = list(full_ids)

            MAX_INPUT = SEQ_LEN - max_new_tokens - 1
            if len(current_ids) > MAX_INPUT:
                current_ids = current_ids[-MAX_INPUT:]
            
            L_prompt = len(current_ids)

            print(f"""
            Effective Context Length: {L_prompt} tokens
            Prompt (Last Sentence): '{last_sentence.strip()}'
            Target Word (for generation): '{target_str}'
            """)
            
            input_ids = jnp.full((1, SEQ_LEN), BOS, dtype=jnp.int32)
            mask = jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)
            input_ids = input_ids.at[0, :L_prompt].set(jnp.array(current_ids))
            mask = mask.at[0, :L_prompt].set(1)
            
            outputs = model.apply({'params': params}, input_ids, attention_mask=mask, ttt_lr_mult=ttt_lr_mult)
            logits = outputs.logits[0, L_prompt-1, :]
            probs = jax.nn.softmax(logits)
            
            top_indices = jnp.argsort(logits)[-10:][::-1]
            
            print(f"\n[Doc2Vec Semantic Alignment with Context]:")
            print(f"{'Rank':<5} | {'Prob':<8} | {'D2V Sim':<8} | {'Token':<15} | {'Thematic Visual'}")
            print("-" * 85)
            
            for rank, idx in enumerate(top_indices, 1):
                token_id = idx.item()
                token_str = tokenizer.decode([token_id]).strip()
                
                candidate_vec = d2v_model.infer_vector(word_tokenize(token_str.lower()))
                
                sim = cosine_similarity(prompt_vec, candidate_vec).item()
                
                p = probs[token_id].item()
                bar = draw_bar(max(0, sim), length=20)
                print(f"{rank:<5} | {p:<8.4f} | {sim:<8.4f} | '{token_str[:13]:<13}' | {bar}")
            
            print("\nGeneration: ", end="", flush=True)
            for step in range(max_new_tokens):
                L = len(current_ids)
                if L >= SEQ_LEN: break
                
                if step > 0:
                    input_ids = jnp.full((1, SEQ_LEN), BOS, dtype=jnp.int32)
                    mask = jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)
                    input_ids = input_ids.at[0, :L].set(jnp.array(current_ids))
                    mask = mask.at[0, :L].set(1)
                    outputs = model.apply({'params': params}, input_ids, attention_mask=mask, ttt_lr_mult=ttt_lr_mult)
                    logits = outputs.logits[0, L-1, :]

                next_token_id = jnp.argmax(logits).item()
                current_ids.append(next_token_id)
                print(tokenizer.decode([next_token_id]), end="", flush=True)
                if next_token_id == tokenizer.eos_token_id: break

            print("\n" + "=" * 80 + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with TTT-LM.")
    parser.add_argument(
        "--transformer_pretrained_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained the Self-Attention based Transformer model.",
    )
    parser.add_argument(
        "--ttt_pretrained_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained TTT model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--ttt_lr_mult",
        type=float,
        default=0.0,
        help="TTT learning rate multiplier.",
        required=False
    )
    return parser.parse_args()

if __name__ == "__main__":
    set_random_seed(42)
    args = parse_args()
    test_cases = [
        {
            "domain": "Calculus",
            "context": "The derivative of a function at a chosen input value describes the rate of change of the function near that input value.",
            "prompt": " The derivative of a function describes the rate of",
            "target": "change of the function near that input value."
        },
        {
            "domain": "Biology",
            "context": "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions.",
            "prompt": " Chemical energy for the cell is produced by the",
            "target": "mitochondria."
        },
        {
            "domain": "Chemistry",
            "context": "The periodic table is a tabular display of the chemical elements, which are arranged by atomic number, electron configuration, and recurring chemical properties.",
            "prompt": " In the periodic table, elements are arranged by their atomic",
            "target": "number, electron configuration, and recurring chemical properties."
        },
        {
            "domain": "Geography",
            "context": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
            "prompt": " The mountain range where Everest is located is the",
            "target": "Mahalangur Himal sub-range of the Himalayas."
        }
    ]
    
    models = [args.selfattention_pretrained_model_path, args.ttt_pretrained_model_path]
    for model_path in models:
        if model_path == args.ttt_pretrained_model_path:
            run_inference(model_path, test_cases, ttt_lr_mult=args.ttt_lr_mult)
        else:
            run_inference(model_path, test_cases)
