import requests

def generate(context, token_max_length=20, temperature=0.1, top_p=1.0):
    assert isinstance(token_max_length, int), "Max token most be integer value"
    assert isinstance(temperature, float), "temperature most be float value"
    assert isinstance(top_p, float), "top_p most be float value"
    payload = {
        "context": str(context),
        "token_max_length": token_max_length,
        "temperature": temperature,
        "top_p": top_p
    }
    URL = requests.post("http://api.vicgalle.net:5000/generate", params=payload)
    text = URL.json()
    return str(text["text"])

dialog = [
    'Human: Hi!',
    'Alpha: Hi!',
    'Human: Who are you?',
    'Alpha: I\'m a friendly chatbot.',
    'Human: What is your name?',
    'Alpha: My name is Alpha.',
    'Human: What is your faviorite book?',
    'Alpha: My favorite book is Harry Potter.',
    'Human: What do you like to eat?',
    'Alpha: I am a chatbot, I don\'t eat.',
]

while True:
    phrase = input('Human: ')
    dialog.append('Human: ' + phrase)
    text = '\n'.join(dialog) + '\nAlpha: '
    # print(repr(text))
    answer = generate(context=text)
    print(repr(answer))
    # answer = answer.strip()
    endl = answer.find('\n')
    if endl != -1:
        answer = answer[:endl]

    dialog.append('Alpha: ' + answer)

    print('Alpha:', answer)