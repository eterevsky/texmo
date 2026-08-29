# ELIZA (vendored)

Joseph Weizenbaum's DOCTOR script, running on a faithful Python
re-implementation of the 1966 pattern matcher.

- Upstream: <https://github.com/wadetb/eliza>
- Commit: `6055a7ded7b9ff08beff75c2d9ec56fb7b0ee639` (2019-09-15)
- License: MIT, `LICENSE` (Copyright (c) 2019 Wade Brainerd)
- Files taken verbatim: `doctor.txt` (the script: keywords, weights,
  decomposition/reassembly rules, pre/post substitutions, synonyms),
  `LICENSE`
- `eliza.py`: upstream's matcher plus two local edits, both listed in
  its module docstring -- an injectable random source and a UTF-8 open
  in `load()`. No behavioural change; no dependencies.

Used as a calibration student in `scripts/chat_eval.py`
(`generate --student-eliza`): a pattern matcher that reflects the
user's own words is "responsive but empty", which brackets the
scripted-examiner eval from a different side than the null phrase bot.

```python
import eliza
bot = eliza.load_doctor()          # or load_doctor(random.Random(0))
bot.respond("I am sad today")
```
