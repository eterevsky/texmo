from gutenbergpy import textget

for i in range(1, 20):
    try:
        raw_text = textget.get_text_by_id(i)
        text = textget.strip_headers(raw_text)
        text = text.decode('UTF-8')
    except Exception:
        continue
    text = text.replace('\r', '')
    with open(f'data/{i}.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'{i}: {len(text)}')
