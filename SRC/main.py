
from transformers import AutoTokenizer

# define the sentence to tokenize
sentence = "Hello world!"

# load the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# apply the tokenizer to the sentence and extract the token ids
token_ids = tokenizer(sentence).input_ids

print(token_ids)

# Note: It's better to decode the whole list at once, like tokenizer.decode(token_ids)
# Or if you must loop, wrap the id in a list to decode one at a time correctly.
for id in token_ids:
    print(tokenizer.decode([id]))

# A list of colors in RGB for representing the tokens
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence: str, tokenizer):
    """ Show the tokens each separated by a different color """

    # Tokenize the input
    token_ids = tokenizer(sentence).input_ids

    # Extract vocabulary length
    print(f"Vocab length:{len(tokenizer)}")

    # Print a colored list of tokens
    for idx, token_id in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode([token_id]) +
            '\x1b[0m',
            end=' '
        )
    print() # Add a newline after printing all tokens
  
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""

# Load tokenizers once to avoid reloading them on every function call
print("\n--- Bert Base Cased ---")
bert_cased_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
show_tokens(text, bert_cased_tokenizer)

print("\n\n--- Bert Base Uncased ---")
bert_uncased_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
show_tokens(text, bert_uncased_tokenizer)

print("\n\n--- GPT-4 (Xenova) ---")
gpt4_tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
show_tokens(text, gpt4_tokenizer)