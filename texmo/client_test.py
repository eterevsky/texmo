"""Tests for the client-side pieces that don't need a server: the
training-data CSV parse (with its spec+precision parse dedup)."""

from texmo.client import parse_training_csv


def test_parse_training_csv_dedups_models():
    text = (
        "spec,precision,lr,length,batch,steps,decay,cosine,loss\n"
        "bytes|dense.32.gelu,fp32,0.1,128,32,256,1.0,0,3.0\n"
        "bytes|dense.32.gelu,fp32,0.05,128,32,512,0.5,1,2.5\n"
        "bits.1+bp|gru.4,fp32,0.1,64,16,256,1.0,0,1.2\n"
    )
    data = parse_training_csv(text)
    assert len(data) == 3
    (c1, l1), (c2, l2), (c3, l3) = data
    assert (l1, l2, l3) == (3.0, 2.5, 1.2)
    # Same (spec, precision) reuses one parsed model object.
    assert c1.model is c2.model
    assert c3.model is not c1.model
    assert c2.cosine is True and c2.decay == 0.5 and c2.steps == 512
    assert c1.model.num_weights == c2.model.num_weights
