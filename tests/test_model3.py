import logging
import os
from unittest import TestCase

from texmo.model3 import build_model
from texmo.tokens import set_tokens_dir

logging.disable(level=logging.ERROR)
set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))


class Model3Test(TestCase):
    def test_build_model(self):
        model = build_model("tokens.128-pos.16-emb.64.norm|gru.64-dense.128.relu")
        self.assertEqual(
            str(model), "tokens.128-pos.16-emb.64.norm|gru.64-dense.128.relu"
        )

    def test_neighbors(self):
        model = build_model("tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu")
        neighbors = set(map(str, model.neighbors()))
        self.assertEqual(
            neighbors,
            {
                "tokens.128-pos.16-emb.128.norm|gru.64-dense.64.relu",
                "tokens.128-pos.16-emb.32.norm|gru.64-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|dense.64.gelu-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|dense.64.relu-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|dense.64.tanh-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|gru.128-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|gru.32-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.128.relu",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.32.relu",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.gelu",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-attn.2.2.64",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-attnmq.2.2.64",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-dense.64.gelu",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-dense.64.tanh",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-gru.64",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-lstm.64",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-mgru.64",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-rec.64.gelu",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-rec.64.relu",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-rec.64.tanh",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.relu-suffix.2",
                "tokens.128-pos.16-emb.64.norm|gru.64-dense.64.tanh",
                "tokens.128-pos.16-emb.64.norm|gru.64-gru.64",
                "tokens.128-pos.16-emb.64.norm|gru.64-lstm.64",
                "tokens.128-pos.16-emb.64.norm|gru.64-mgru.64",
                "tokens.128-pos.16-emb.64.norm|gru.64-rec.64.gelu",
                "tokens.128-pos.16-emb.64.norm|gru.64-rec.64.relu",
                "tokens.128-pos.16-emb.64.norm|gru.64-rec.64.tanh",
                "tokens.128-pos.16-emb.64.norm|gru.64",
                "tokens.128-pos.16-emb.64.norm|lstm.64-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|mgru.64-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|rec.64.gelu-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|rec.64.relu-dense.64.relu",
                "tokens.128-pos.16-emb.64.norm|rec.64.tanh-dense.64.relu",
                "tokens.128-pos.16-emb.64|gru.64-dense.64.relu",
                "tokens.128-pos.32-emb.64.norm|gru.64-dense.64.relu",
                "tokens.128-pos.8-emb.64.norm|gru.64-dense.64.relu",
                "tokens.256-pos.16-emb.64.norm|gru.64-dense.64.relu",
                "tokens.64-pos.16-emb.64.norm|gru.64-dense.64.relu",
            },
        )
