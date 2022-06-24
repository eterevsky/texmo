from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

for i in range(101, 1001):
    try:
        text = strip_headers(load_etext(i)).strip()
    except Exception:
        continue
    text = text.replace('\r', '')
    with open(f'data/{i}.txt', 'w') as f:
        f.write(text)
    print(f'{i}: {len(text)}')