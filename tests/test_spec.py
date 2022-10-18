from unittest import main, TestCase

from texmo.spec import ModelSpec, is_reachable_spec


def m(spec):
    return ModelSpec.parse(spec)


class ModelSpecTest(TestCase):
    def test_is_reachable_spec(self):
        init1 = ModelSpec.parse("dense.1.relu")
        self.assertTrue(
            is_reachable_spec(
                init1,
                m("suffix.2-dense.1.relu-suffix.4"),
                ("suffix", "layer", "type", "size"),
            )
        )
        self.assertFalse(
            is_reachable_spec(
                init1,
                m("suffix.2-dense.1.relu-suffix.4"),
                ("layer", "type", "size"),
            )
        )

        self.assertTrue(
            is_reachable_spec(
                init1,
                m("suffix.2-dense.1.relu-suffix.4-rec.32.tanh"),
                ("suffix", "layer", "type", "size"),
            )
        )
        self.assertFalse(
            is_reachable_spec(
                init1,
                m("suffix.2-dense.1.relu-suffix.4-rec.32.tanh"),
                ("suffix", "type", "size"),
            )
        )

        self.assertTrue(
            is_reachable_spec(
                init1,
                m("suffix.2-lstm.32-suffix.4"),
                ("suffix", "type", "size"),
            )
        )
        self.assertTrue(
            is_reachable_spec(
                init1,
                m("suffix.2-dense.32.tanh-suffix.4"),
                ("suffix", "type", "size"),
            )
        )
        self.assertFalse(
            is_reachable_spec(
                init1, m("suffix.2-lstm.32-suffix.4"), ("suffix", "size")
            )
        )
        self.assertFalse(
            is_reachable_spec(
                init1, m("suffix.2-dense.32.tanh-suffix.4"), ("suffix", "size")
            )
        )

        self.assertTrue(
            is_reachable_spec(
                init1, m("suffix.2-dense.32.relu-suffix.4"), ("suffix", "size")
            )
        )
        self.assertFalse(
            is_reachable_spec(
                init1, m("suffix.2-dense.32.relu-suffix.4"), ("suffix",)
            )
        )


if __name__ == "__main__":
    main()
