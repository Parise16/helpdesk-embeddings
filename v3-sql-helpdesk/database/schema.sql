PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    department_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    department_id INTEGER NOT NULL,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE IF NOT EXISTS classification_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'seed',
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    UNIQUE (category_id, text),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE TABLE IF NOT EXISTS category_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER NOT NULL,
    keyword TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0 CHECK (weight >= 0),
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    UNIQUE (category_id, keyword),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requester_name TEXT NOT NULL,
    requester_email TEXT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'TRIAGED', 'IN_PROGRESS', 'RESOLVED')),
    priority TEXT NOT NULL DEFAULT 'MEDIUM' CHECK (priority IN ('LOW', 'MEDIUM', 'HIGH')),
    assigned_department_id INTEGER,
    assigned_employee_id INTEGER,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (assigned_department_id) REFERENCES departments(id),
    FOREIGN KEY (assigned_employee_id) REFERENCES employees(id)
);

CREATE TABLE IF NOT EXISTS ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER NOT NULL UNIQUE,
    predicted_category_id INTEGER NOT NULL,
    semantic_score REAL NOT NULL,
    nlp_score REAL NOT NULL,
    hybrid_score REAL NOT NULL,
    margin REAL NOT NULL,
    decision_source TEXT NOT NULL,
    qwen_used INTEGER NOT NULL DEFAULT 0 CHECK (qwen_used IN (0, 1)),
    needs_review INTEGER NOT NULL DEFAULT 0 CHECK (needs_review IN (0, 1)),
    problem_summary TEXT NOT NULL,
    extracted_location TEXT,
    extracted_entities_json TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    llm_model TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE CASCADE,
    FOREIGN KEY (predicted_category_id) REFERENCES categories(id)
);

CREATE TABLE IF NOT EXISTS ai_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
    corrected_category_id INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES ai_predictions(id) ON DELETE CASCADE,
    FOREIGN KEY (corrected_category_id) REFERENCES categories(id)
);

CREATE TABLE IF NOT EXISTS ticket_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_employee_status ON tickets(assigned_employee_id, status);
CREATE INDEX IF NOT EXISTS idx_tickets_department_status ON tickets(assigned_department_id, status);
CREATE INDEX IF NOT EXISTS idx_examples_category_active ON classification_examples(category_id, active);
CREATE INDEX IF NOT EXISTS idx_keywords_category_active ON category_keywords(category_id, active);
CREATE INDEX IF NOT EXISTS idx_predictions_category ON ai_predictions(predicted_category_id);
CREATE INDEX IF NOT EXISTS idx_history_ticket ON ticket_history(ticket_id, created_at);

CREATE VIEW IF NOT EXISTS v_ticket_overview AS
SELECT
    t.id AS ticket_id,
    t.requester_name,
    t.title,
    t.status,
    t.priority,
    t.created_at,
    t.resolved_at,
    d.name AS department,
    e.name AS employee,
    c.name AS predicted_category,
    p.semantic_score,
    p.nlp_score,
    p.hybrid_score,
    p.margin,
    p.decision_source,
    p.qwen_used,
    p.needs_review,
    p.problem_summary,
    p.extracted_location
FROM tickets t
LEFT JOIN departments d ON d.id = t.assigned_department_id
LEFT JOIN employees e ON e.id = t.assigned_employee_id
LEFT JOIN ai_predictions p ON p.ticket_id = t.id
LEFT JOIN categories c ON c.id = p.predicted_category_id;

CREATE VIEW IF NOT EXISTS v_open_ticket_queue AS
SELECT
    t.id AS ticket_id,
    t.title,
    t.priority,
    t.status,
    d.name AS department,
    e.name AS employee,
    t.created_at
FROM tickets t
LEFT JOIN departments d ON d.id = t.assigned_department_id
LEFT JOIN employees e ON e.id = t.assigned_employee_id
WHERE t.status IN ('OPEN', 'TRIAGED', 'IN_PROGRESS')
ORDER BY t.created_at, t.id;

CREATE VIEW IF NOT EXISTS v_ai_quality AS
SELECT
    p.id AS prediction_id,
    p.ticket_id,
    c.name AS predicted_category,
    p.semantic_score,
    p.nlp_score,
    p.hybrid_score,
    p.margin,
    p.decision_source,
    p.qwen_used,
    p.needs_review,
    f.is_correct,
    cc.name AS corrected_category,
    f.notes,
    f.created_at AS feedback_created_at
FROM ai_predictions p
JOIN categories c ON c.id = p.predicted_category_id
LEFT JOIN ai_feedback f ON f.prediction_id = p.id
LEFT JOIN categories cc ON cc.id = f.corrected_category_id;
