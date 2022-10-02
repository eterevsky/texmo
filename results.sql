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

    -- Cached self-score (median of run scores for this conf)
    score REAL,
    -- Cached, should be invalidated for a new run with different `vary`.
    cluster_score REAL
);

CREATE INDEX conf_spec ON conf(spec);

CREATE TABLE neighbor (
    conf1_id INTEGER NOT NULL,
    conf2_id INTEGER NOT NULL,
    vary INTEGER NOT NULL,
    FOREIGN KEY (conf1_id) REFERENCES conf(id),
    FOREIGN KEY (conf2_id) REFERENCES conf(id)
);

CREATE INDEX neighbor_conf1_id ON neighbor(conf1_id);

CREATE TABLE run (
    conf_id INTEGER NOT NULL,

    timestamp TEXT,
    test_sample_len INTEGER,
    test_batch INTEGER,
    loss REAL NOT NULL
);