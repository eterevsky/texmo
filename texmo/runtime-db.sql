CREATE TABLE conf (
    id INTEGER NOT NULL PRIMARY KEY,

    weights INTEGER NOT NULL,
    matches_template BOOLEAN NOT NULL,

    -- Median time on the current system.
    median_time REAL,
    estimated_time REAL,
    -- Cached self-score (median of run scores for this conf)
    median_score REAL,
    neighbors_score REAL
    -- Predicted score from the Predictor model
    -- pred_score REAL
);

CREATE INDEX conf_neighbors_score ON conf(neighbors_score);
-- CREATE INDEX conf_pred_score ON conf(pred_score);

-- CREATE TABLE neighbor (
--     conf1_id INTEGER NOT NULL,
--     conf2_id INTEGER NOT NULL,
--     FOREIGN KEY (conf1_id) REFERENCES conf(id),
--     FOREIGN KEY (conf2_id) REFERENCES conf(id)
-- );

-- CREATE INDEX neighbor_conf1_id ON neighbor(conf1_id);
-- CREATE INDEX neighbor_conf2_id ON neighbor(conf2_id);
 