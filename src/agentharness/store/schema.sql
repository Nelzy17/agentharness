-- Two tables. A run, and the observable steps it took.
--
-- What is absent is deliberate. There is no column for reasoning, rationale or
-- explanation, and none for the assembled context. The trace records what was
-- requested and what happened, not what the model was thinking and not the
-- transcript it was thinking about. Storing the messages "for debugging" is the
-- change that turns a trace into a transcript, and a schema test asserts the
-- column set so that change has to be made on purpose.

CREATE TABLE IF NOT EXISTS runs (
    run_id                   TEXT PRIMARY KEY,
    goal                     TEXT    NOT NULL,
    model                    TEXT    NOT NULL,
    terminal_reason          TEXT,
    -- The human-readable half of the reason, stored rather than derived, so a
    -- trace is readable as data without importing the enum that produced it.
    reason_text              TEXT,
    route                    TEXT,
    answer                   TEXT,
    sources                  TEXT    NOT NULL DEFAULT '[]',
    insufficient_information INTEGER NOT NULL DEFAULT 0,
    iterations               INTEGER NOT NULL DEFAULT 0,
    prompt_tokens            INTEGER NOT NULL DEFAULT 0,
    completion_tokens        INTEGER NOT NULL DEFAULT 0,
    cached_tokens            INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens       INTEGER NOT NULL DEFAULT 0,
    duration_ms              REAL,
    -- Full detail of a harness-fatal failure. The caller gets a run id; the
    -- detail stays here.
    error                    TEXT,
    started_at               REAL    NOT NULL,
    ended_at                 REAL
);

CREATE TABLE IF NOT EXISTS steps (
    step_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT    NOT NULL REFERENCES runs(run_id),
    sequence           INTEGER NOT NULL,
    iteration          INTEGER NOT NULL,
    kind               TEXT    NOT NULL CHECK (kind IN ('model_call', 'tool_call')),
    duration_ms        REAL    NOT NULL,

    -- model_call
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    cached_tokens      INTEGER,
    cache_write_tokens INTEGER,
    attempts           INTEGER,

    -- tool_call
    tool_call_id       TEXT,
    tool_name          TEXT,
    arguments_json     TEXT,
    outcome_class      TEXT,
    -- Written now though M6 owns write policy: adding a column to a table with
    -- rows in it is worse than carrying an unused one.
    is_write           INTEGER,
    -- The other half of the sanitized/full split. The model got a generic
    -- message; the traceback lands here and nowhere the model can read.
    error_detail       TEXT
);

CREATE INDEX IF NOT EXISTS steps_by_run ON steps (run_id, sequence);
