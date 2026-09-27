import tiktoken
enc = tiktoken.get_encoding("gpt2")
with open("math_train_curriculum.txt", "r") as f:
    text = f.read()
blocks = text.split("\n\n")
max_len = 0
for b in blocks:
    l = len(enc.encode(b))
    if l > max_len: max_len = l
print("Max tokens in a single problem:", max_len)
