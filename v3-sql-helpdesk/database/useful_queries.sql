SELECT * FROM v_ticket_overview ORDER BY ticket_id DESC;

SELECT * FROM v_open_ticket_queue;

SELECT
    e.name AS employee,
    d.name AS department,
    COUNT(t.id) AS open_tickets
FROM employees e
JOIN departments d ON d.id = e.department_id
LEFT JOIN tickets t
    ON t.assigned_employee_id = e.id
   AND t.status IN ('OPEN', 'TRIAGED', 'IN_PROGRESS')
WHERE e.active = 1
GROUP BY e.id, e.name, d.name
ORDER BY d.name, open_tickets, e.name;

SELECT
    c.name AS category,
    COUNT(p.id) AS predictions,
    ROUND(AVG(p.semantic_score), 3) AS avg_semantic,
    ROUND(AVG(p.nlp_score), 3) AS avg_nlp,
    ROUND(AVG(p.hybrid_score), 3) AS avg_hybrid
FROM categories c
LEFT JOIN ai_predictions p ON p.predicted_category_id = c.id
GROUP BY c.id, c.name
ORDER BY predictions DESC;

SELECT
    COUNT(*) AS total_predictions,
    SUM(qwen_used) AS qwen_calls,
    ROUND(100.0 * SUM(qwen_used) / NULLIF(COUNT(*), 0), 1) AS qwen_rate_percent
FROM ai_predictions;

SELECT
    c.name AS category,
    ce.text,
    ce.source,
    ce.active
FROM classification_examples ce
JOIN categories c ON c.id = ce.category_id
ORDER BY c.name, ce.id;

SELECT
    c.name AS category,
    ck.keyword,
    ck.weight
FROM category_keywords ck
JOIN categories c ON c.id = ck.category_id
WHERE ck.active = 1
ORDER BY c.name, ck.weight DESC, ck.keyword;

SELECT * FROM v_ai_quality ORDER BY prediction_id DESC;


-- Previsões ainda sem revisão humana
SELECT
    p.id AS prediction_id,
    t.id AS ticket_id,
    t.title,
    t.description,
    c.name AS predicted_category,
    p.semantic_score,
    p.nlp_score,
    p.hybrid_score,
    p.margin,
    p.qwen_used,
    p.decision_source
FROM ai_predictions p
JOIN tickets t ON t.id = p.ticket_id
JOIN categories c ON c.id = p.predicted_category_id
LEFT JOIN ai_feedback f ON f.prediction_id = p.id
WHERE f.id IS NULL
ORDER BY p.id;

-- Resultado das revisões humanas
SELECT *
FROM v_ai_quality
WHERE is_correct IS NOT NULL
ORDER BY prediction_id;
