CREATE TABLE conf (
    id INTEGER NOT NULL PRIMARY KEY,

    spec TEXT NOT NULL,
    lr REAL NOT NULL,
    sample_len INTEGER NOT NULL,
    batch INTEGER NOT NULL,
    regularization REAL NOT NULL,
    init_scale REAL NOT NULL,
    t INTEGER NOT NULL,
    weights INTEGER NOT NULL,
    cluster_score REAL
);

CREATE TABLE run (
    conf_id INTEGER NOT NULL,

    timestamp TEXT,
    test_sample_len INTEGER,
    test_batch INTEGER,
    loss REAL NOT NULL,
    FOREIGN KEY (conf_id) REFERENCES conf(id)
);

CREATE INDEX run_conf_id ON run(conf_id);
