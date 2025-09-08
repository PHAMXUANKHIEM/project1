-- Initialize AI Monitoring Database
CREATE TABLE IF NOT EXISTS metrics_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2), 
    disk_usage DECIMAL(5,2),
    network_traffic BIGINT,
    load_average DECIMAL(5,2),
    health_score DECIMAL(5,2),
    failure_probability DECIMAL(5,4),
    is_anomaly BOOLEAN DEFAULT FALSE,
    predictions JSONB,
    raw_metrics JSONB
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_health ON metrics_history(health_score);
CREATE INDEX IF NOT EXISTS idx_metrics_failure ON metrics_history(failure_probability);

CREATE TABLE IF NOT EXISTS alerts_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    alert_name VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    description TEXT,
    state VARCHAR(50) NOT NULL,
    metrics JSONB,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts_history(severity);

-- Insert sample configuration
INSERT INTO metrics_history (cpu_usage, memory_usage, disk_usage, network_traffic, load_average, health_score, failure_probability) 
VALUES (25.5, 45.2, 32.1, 1024000, 1.2, 85.0, 0.1) ON CONFLICT DO NOTHING;

COMMIT;
