CREATE TABLE conf (
    id INTEGER NOT NULL PRIMARY KEY,

    spec TEXT NOT NULL,
    lr REAL NOT NULL,
    sample_len INTEGER NOT NULL,
    batch INTEGER NOT NULL,
    t INTEGER NOT NULL,
    weights INTEGER NOT NULL,
    cluster_score REAL
);

CREATE INDEX conf_spec ON conf(spec);

CREATE TABLE run (
    id INTEGER NOT NULL PRIMARY KEY,

    conf_id INTEGER NOT NULL,

    timestamp TEXT,
    test_sample_len INTEGER,
    test_batch INTEGER,
    loss REAL NOT NULL,
    -- A 1D array of training losses after each step, encoded as
    -- ndarray(dtype=np.float32) and converted to bytes by ndarray.tobytes().
    -- Can be converted back to an array by np.frombuffer().
    step_loss BLOB,
    -- Loss model version.
    loss_model_v INTEGER,
    -- Parameters of loss model as ndarray(dtype=np.float32).
    loss_model BLOB,
    FOREIGN KEY (conf_id) REFERENCES conf(id)
);

CREATE INDEX run_conf_id ON run(conf_id);
