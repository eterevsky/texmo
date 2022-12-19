CREATE TABLE conf (
    id INTEGER NOT NULL PRIMARY KEY,

    spec TEXT NOT NULL,
    lr REAL NOT NULL,
    sample_len INTEGER NOT NULL,
    batch INTEGER NOT NULL,
    t INTEGER NOT NULL,
    weights INTEGER NOT NULL,

    -- Cached self-score (median of run scores for this conf)
    score REAL,
    -- Predicted score from the Predictor model
    pred_score REAL
);

CREATE INDEX conf_spec ON conf(spec);
CREATE INDEX conf_score ON conf(score);
CREATE INDEX conf_pred_score ON conf(pred_score);

CREATE TABLE neighbor (
    conf1_id INTEGER NOT NULL,
    conf2_id INTEGER NOT NULL,
    FOREIGN KEY (conf1_id) REFERENCES conf(id),
    FOREIGN KEY (conf2_id) REFERENCES conf(id)
);

CREATE INDEX neighbor_conf1_id ON neighbor(conf1_id);
CREATE INDEX neighbor_conf2_id ON neighbor(conf2_id);

CREATE TABLE run (
    conf_id INTEGER NOT NULL,
    loss REAL NOT NULL,
    FOREIGN KEY (conf_id) REFERENCES conf(id)
);

CREATE INDEX run_conf_id ON run(conf_id);
