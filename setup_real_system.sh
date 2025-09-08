#!/bin/bash
# setup_real_system.sh - Setup AI monitoring với Prometheus/Grafana thật

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROMETHEUS_HOST="54.179.129.166"
PROMETHEUS_PORT="9090"
GRAFANA_HOST="54.179.129.166"
GRAFANA_PORT="3000"

PROMETHEUS_URL="http://${PROMETHEUS_HOST}:${PROMETHEUS_PORT}"
GRAFANA_URL="http://${GRAFANA_HOST}:${GRAFANA_PORT}"

echo -e "${BLUE}🚀 Setting up AI Monitoring System${NC}"
echo -e "${BLUE}   Prometheus: ${PROMETHEUS_URL}${NC}"
echo -e "${BLUE}   Grafana: ${GRAFANA_URL}${NC}"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_status "Docker is installed"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_status "Docker Compose is installed"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi
print_status "Python 3 is installed"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi
print_status "Node.js is installed"

# Check connectivity to Prometheus
echo -e "${BLUE}🔗 Testing connectivity to existing Prometheus/Grafana...${NC}"

if curl -s --connect-timeout 10 "${PROMETHEUS_URL}/-/healthy" > /dev/null; then
    print_status "Successfully connected to Prometheus at ${PROMETHEUS_URL}"
else
    print_error "Cannot connect to Prometheus at ${PROMETHEUS_URL}"
    print_warning "Please check if Prometheus is running and accessible"
    exit 1
fi

if curl -s --connect-timeout 10 "${GRAFANA_URL}/api/health" > /dev/null; then
    print_status "Successfully connected to Grafana at ${GRAFANA_URL}"
else
    print_warning "Cannot connect to Grafana at ${GRAFANA_URL} (this is optional)"
fi

# Create project structure
echo -e "${BLUE}📁 Creating project structure...${NC}"
mkdir -p {backend,frontend,scripts,config,logs,data,models}
print_status "Project directories created"

# Create environment file
echo -e "${BLUE}⚙️  Creating configuration files...${NC}"

cat > .env << EOF
# Prometheus/Grafana Configuration
PROMETHEUS_URL=${PROMETHEUS_URL}
GRAFANA_URL=${GRAFANA_URL}

# AI Service Configuration
AI_SERVICE_PORT=8000
API_SERVICE_PORT=8081
DASHBOARD_PORT=3001

# Database Configuration
POSTGRES_DB=ai_monitoring
POSTGRES_USER=ai_user
POSTGRES_PASSWORD=ai_password_$(openssl rand -hex 8)

# Redis Configuration
REDIS_PASSWORD=redis_password_$(openssl rand -hex 8)

# Monitoring Configuration
MONITORING_INTERVAL=30
LOG_LEVEL=INFO
MODEL_RETRAIN_INTERVAL=3600

# Email Alerts (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
# SMTP_USERNAME=your-email@gmail.com
# SMTP_PASSWORD=your-app-password

# Slack Alerts (optional)
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
EOF

print_status "Environment configuration created"

# Create requirements.txt
cat > requirements.txt << EOF
# Core dependencies
aiohttp==3.8.5
fastapi==0.104.1
uvicorn==0.24.0
asyncio

# Data processing
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# Monitoring
prometheus-client==0.17.1

# Database
psycopg2-binary==2.9.7
sqlalchemy==2.0.23
asyncpg==0.28.0
alembic==1.12.1

# Caching
redis==5.0.1

# Utils
python-dotenv==1.0.0
python-multipart==0.0.6
jinja2==3.1.2

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1

# Code quality
flake8==6.0.0
black==23.9.1
bandit==1.7.5
safety==2.3.4
EOF

print_status "Python requirements created"

# Download and setup AI monitoring service
echo -e "${BLUE}🤖 Setting up AI monitoring service...${NC}"

# Create the main AI service file
cat > backend/ai_monitoring_service.py <<'EOF'
# AI Monitoring Service - Production Ready
import asyncio
import aiohttp
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import logging
import json
from datetime import datetime, timedelta
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# Prometheus metrics
cpu_usage_gauge = Gauge('ai_cpu_usage_percent', 'AI analyzed CPU usage')
memory_usage_gauge = Gauge('ai_memory_usage_percent', 'AI analyzed Memory usage')
disk_usage_gauge = Gauge('ai_disk_usage_percent', 'AI analyzed Disk usage')
network_traffic_gauge = Gauge('ai_network_traffic_bytes_per_sec', 'AI analyzed Network traffic')
prediction_gauge = Gauge('predicted_system_failure_probability', 'AI system failure prediction')
health_score_gauge = Gauge('system_health_score', 'Overall system health (0-100)')
anomaly_counter = Counter('anomalies_detected_total', 'Total anomalies detected')
prediction_histogram = Histogram('prediction_duration_seconds', 'Time spent on predictions')

class AIMonitoringService:
    def __init__(self):
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://54.179.129.166:9090')
        self.monitoring_interval = int(os.getenv('MONITORING_INTERVAL', 30))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # AI Model setup
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        self.historical_data = []
        self.model_path = Path('models/anomaly_detector.pkl')
        self.scaler_path = Path('models/scaler.pkl')
        
        # Logging setup
        self.logger = self._setup_logging()
        self.session = None
        
        # Load existing model if available
        self._load_models()
    
    def _setup_logging(self):
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=log_format,
            handlers=[
                logging.FileHandler('logs/ai_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.anomaly_detector = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                self.logger.info("🤖 Loaded existing AI models")
        except Exception as e:
            self.logger.warning(f"Could not load existing models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            self.model_path.parent.mkdir(exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.anomaly_detector, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.logger.info("💾 Saved AI models")
        except Exception as e:
            self.logger.error(f"Could not save models: {e}")
    
    async def fetch_metrics(self):
        """Fetch metrics from real Prometheus"""
        queries = {
            'cpu': 'avg(100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))',
            'memory': 'avg((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100)',
            'disk': 'avg((1 - (node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"})) * 100)',
            'network_in': 'avg(irate(node_network_receive_bytes_total{device!="lo"}[5m]))',
            'network_out': 'avg(irate(node_network_transmit_bytes_total{device!="lo"}[5m]))',
            'load1': 'avg(node_load1)',
            'load5': 'avg(node_load5)',
            'load15': 'avg(node_load15)',
        }
        
        metrics = {}
        try:
            for name, query in queries.items():
                url = f"{self.prometheus_url}/api/v1/query"
                params = {'query': query}
                
                async with self.session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == 'success' and data['data']['result']:
                            value = float(data['data']['result'][0]['value'][1])
                            metrics[name] = max(0, value)  # Ensure non-negative
                        else:
                            metrics[name] = 0.0
                    else:
                        self.logger.warning(f"HTTP {response.status} for {name}")
                        metrics[name] = 0.0
            
            # Calculate total network traffic
            metrics['network'] = metrics.get('network_in', 0) + metrics.get('network_out', 0)
            
        except Exception as e:
            self.logger.error(f"Error fetching metrics: {e}")
            # Return safe defaults
            metrics = {k: 0.0 for k in queries.keys()}
            metrics['network'] = 0.0
        
        return metrics
    
    def calculate_health_score(self, metrics):
        """Calculate system health score (0-100)"""
        score = 100.0
        
        # CPU penalties
        cpu = metrics.get('cpu', 0)
        if cpu > 95: score -= 40
        elif cpu > 85: score -= 25
        elif cpu > 75: score -= 15
        elif cpu > 65: score -= 8
        
        # Memory penalties
        memory = metrics.get('memory', 0)
        if memory > 98: score -= 35
        elif memory > 90: score -= 20
        elif memory > 80: score -= 12
        elif memory > 70: score -= 6
        
        # Disk penalties
        disk = metrics.get('disk', 0)
        if disk > 98: score -= 30
        elif disk > 95: score -= 20
        elif disk > 85: score -= 10
        elif disk > 75: score -= 5
        
        # Load average penalties (assuming 4-core system)
        load1 = metrics.get('load1', 0)
        if load1 > 8: score -= 20
        elif load1 > 6: score -= 12
        elif load1 > 4: score -= 6
        
        return max(0, min(100, score))
    
    def train_model(self):
        """Train anomaly detection model"""
        if len(self.historical_data) < 50:
            return False
        
        try:
            df = pd.DataFrame(self.historical_data)
            features = ['cpu', 'memory', 'disk', 'network', 'load1']
            
            if not all(f in df.columns for f in features):
                self.logger.warning("Missing features for training")
                return False
            
            X = df[features].values
            X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)
            
            if X.shape[0] < 50:
                return False
            
            X_scaled = self.scaler.fit_transform(X)
            self.anomaly_detector.fit(X_scaled)
            self.is_trained = True
            
            # Save models
            self._save_models()
            
            self.logger.info(f"🤖 AI model trained with {len(self.historical_data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return False
    
    @prediction_histogram.time()
    def predict_failure(self, current_metrics):
        """Predict system failure probability"""
        with prediction_histogram.time():
            failure_prob = 0.0
            
            try:
                # Current critical levels
                cpu = current_metrics.get('cpu', 0)
                memory = current_metrics.get('memory', 0)
                disk = current_metrics.get('disk', 0)
                load1 = current_metrics.get('load1', 0)
                
                # Immediate danger levels
                if cpu > 98: failure_prob += 0.5
                elif cpu > 90: failure_prob += 0.3
                elif cpu > 80: failure_prob += 0.15
                
                if memory > 98: failure_prob += 0.4
                elif memory > 95: failure_prob += 0.25
                elif memory > 85: failure_prob += 0.1
                
                if disk > 99: failure_prob += 0.6
                elif disk > 95: failure_prob += 0.3
                elif disk > 90: failure_prob += 0.15
                
                if load1 > 10: failure_prob += 0.3
                elif load1 > 8: failure_prob += 0.2
                elif load1 > 6: failure_prob += 0.1
                
                # Trend analysis if we have enough data
                if len(self.historical_data) >= 10:
                    recent = self.historical_data[-10:]
                    cpu_trend = np.polyfit(range(10), [d.get('cpu', 0) for d in recent], 1)[0]
                    mem_trend = np.polyfit(range(10), [d.get('memory', 0) for d in recent], 1)[0]
                    
                    if cpu_trend > 2: failure_prob += 0.2  # Rising >2%/interval
                    if mem_trend > 1.5: failure_prob += 0.25  # Rising >1.5%/interval
                
                # Anomaly detection bonus
                if self.is_trained:
                    features = np.array([[cpu, memory, disk, current_metrics.get('network', 0), load1]])
                    features = np.nan_to_num(features, nan=0.0)
                    
                    try:
                        features_scaled = self.scaler.transform(features)
                        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
                        if is_anomaly:
                            failure_prob += 0.2
                            anomaly_counter.inc()
                    except Exception as e:
                        self.logger.debug(f"Anomaly detection error: {e}")
                
                # Multiple high resources
                high_count = sum([cpu > 80, memory > 80, disk > 90, load1 > 6])
                if high_count >= 3: failure_prob += 0.3
                elif high_count >= 2: failure_prob += 0.15
                
            except Exception as e:
                self.logger.error(f"Error in failure prediction: {e}")
                return 0.1
            
            return min(1.0, max(0.0, failure_prob))
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        self.session = aiohttp.ClientSession()
        retrain_counter = 0
        
        try:
            while True:
                try:
                    # Fetch metrics
                    metrics = await self.fetch_metrics()
                    metrics['timestamp'] = datetime.now()
                    
                    # Store history
                    self.historical_data.append(metrics)
                    if len(self.historical_data) > 2000:  # Keep last 2000 samples
                        self.historical_data = self.historical_data[-1500:]  # Trim to 1500
                    
                    # Train model periodically
                    retrain_counter += 1
                    if not self.is_trained and len(self.historical_data) >= 50:
                        self.train_model()
                    elif retrain_counter >= 120 and len(self.historical_data) >= 100:  # Retrain every ~1 hour
                        self.train_model()
                        retrain_counter = 0
                    
                    # AI analysis
                    failure_prob = self.predict_failure(metrics)
                    health_score = self.calculate_health_score(metrics)
                    
                    # Update Prometheus metrics
                    cpu_usage_gauge.set(metrics.get('cpu', 0))
                    memory_usage_gauge.set(metrics.get('memory', 0))
                    disk_usage_gauge.set(metrics.get('disk', 0))
                    network_traffic_gauge.set(metrics.get('network', 0))
                    prediction_gauge.set(failure_prob)
                    health_score_gauge.set(health_score)
                    
                    # Enhanced logging with status icons
                    if failure_prob > 0.8:
                        status = "🚨 CRITICAL"
                        level = "critical"
                    elif failure_prob > 0.6:
                        status = "⚠️  WARNING"
                        level = "warning"
                    elif failure_prob > 0.3:
                        status = "💛 CAUTION"
                        level = "info"
                    else:
                        status = "✅ NORMAL"
                        level = "info"
                    
                    log_msg = (
                        f"{status} | "
                        f"CPU: {metrics.get('cpu', 0):.1f}% | "
                        f"RAM: {metrics.get('memory', 0):.1f}% | "
                        f"Disk: {metrics.get('disk', 0):.1f}% | "
                        f"Load: {metrics.get('load1', 0):.2f} | "
                        f"Health: {health_score:.0f}/100 | "
                        f"Risk: {failure_prob:.3f}"
                    )
                    
                    if level == "critical":
                        self.logger.critical(log_msg)
                    elif level == "warning":
                        self.logger.warning(log_msg)
                    else:
                        self.logger.info(log_msg)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                
                await asyncio.sleep(self.monitoring_interval)
                
        finally:
            if self.session:
                await self.session.close()

async def main():
    print("🚀 Starting AI Monitoring Service (Production)")
    print(f"📊 Prometheus: {os.getenv('PROMETHEUS_URL')}")
    print(f"⏱️  Interval: {os.getenv('MONITORING_INTERVAL', 30)}s")
    
    # Start metrics server
    metrics_port = int(os.getenv('AI_SERVICE_PORT', 8000))
    start_http_server(metrics_port)
    print(f"✅ Metrics server started on port {metrics_port}")
    
    # Start monitoring
    service = AIMonitoringService()
    print("🤖 AI features: Anomaly Detection, Failure Prediction, Health Scoring")
    
    await service.monitor_loop()

if __name__ == "__main__":
    asyncio.run(main())
EOF

print_status "AI monitoring service created"

# Setup Python environment
echo -e "${BLUE}🐍 Setting up Python environment...${NC}"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

print_status "Python environment configured"

# Create frontend package.json
echo -e "${BLUE}📦 Setting up React frontend...${NC}"

cat > frontend/package.json << 'EOF'
{
  "name": "ai-monitoring-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^5.16.4",
    "@testing-library/react": "^13.3.0",
    "@testing-library/user-event": "^13.5.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "recharts": "^2.8.0",
    "lucide-react": "^0.263.1",
    "axios": "^1.5.0",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "lint": "eslint src/"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://localhost:8080"
}
EOF

# Install frontend dependencies
cd frontend
npm install
cd ..

print_status "Frontend dependencies installed"

# Create Docker Compose file
echo -e "${BLUE}🐳 Creating Docker configuration...${NC}"

cat > docker-compose.yml << EOF

services:
  ai-monitoring-service:
    build:
      context: .
      dockerfile: Dockerfile.ai
    container_name: ai-monitoring-service
    ports:
      - "8000:8000"  # Metrics
      - "8081:8081"  # API
    environment:
      - PROMETHEUS_URL=${PROMETHEUS_URL}
      - GRAFANA_URL=${GRAFANA_URL}
      - MONITORING_INTERVAL=30
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
    depends_on:
      - redis
      - postgres
    networks:
      - ai-monitoring

  redis:
    image: redis:7-alpine
    container_name: ai-monitoring-redis
    command: redis-server --requirepass \${REDIS_PASSWORD}
    environment:
      - REDIS_PASSWORD=\${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - ai-monitoring

  postgres:
    image: postgres:15
    container_name: ai-monitoring-db
    environment:
      - POSTGRES_DB=\${POSTGRES_DB}
      - POSTGRES_USER=\${POSTGRES_USER}
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - ai-monitoring

volumes:
  redis_data:
  postgres_data:

networks:
  ai-monitoring:
    driver: bridge
EOF

# Create Dockerfile for AI service
cat > Dockerfile.ai << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./
COPY .env .

# Create directories
RUN mkdir -p logs models data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
  CMD curl -f http://localhost:8000/metrics || exit 1

EXPOSE 8000 8081

CMD ["python", "ai_monitoring_service.py"]
EOF

print_status "Docker configuration created"

# Create database init script
mkdir -p scripts
cat > scripts/init.sql << 'EOF'
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
EOF

print_status "Database initialization script created"

# Create startup script
cat > scripts/start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting AI Monitoring System..."

# Load environment
source .env

# Check Prometheus connectivity
echo "📡 Testing Prometheus connection..."
if ! curl -s --connect-timeout 5 "${PROMETHEUS_URL}/-/healthy" > /dev/null; then
    echo "❌ Cannot connect to Prometheus at ${PROMETHEUS_URL}"
    echo "Please ensure Prometheus is running and accessible"
    exit 1
fi
echo "✅ Prometheus connection OK"

# Start services
echo "🐳 Starting Docker containers..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Health checks
echo "🏥 Performing health checks..."

# Check AI service
if curl -s http://localhost:8000/metrics > /dev/null; then
    echo "✅ AI Monitoring Service is healthy"
else
    echo "❌ AI Monitoring Service is not responding"
    docker-compose logs ai-monitoring-service
fi

# Check database
if docker-compose exec -T postgres pg_isready -U ${POSTGRES_USER} > /dev/null; then
    echo "✅ Database is healthy"
else
    echo "❌ Database is not responding"
fi

echo ""
echo "🎉 AI Monitoring System started successfully!"
echo ""
echo "📊 Services:"
echo "   AI Metrics: http://localhost:8000/metrics"
echo "   API: http://localhost:8080/health"
echo "   Database: localhost:5432"
echo ""
echo "🔗 External Services:"
echo "   Prometheus: ${PROMETHEUS_URL}"
echo "   Grafana: ${GRAFANA_URL}"
echo ""
echo "📋 Management Commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop system: docker-compose down"
echo "   Restart: docker-compose restart"
EOF

chmod +x scripts/start.sh
print_status "Startup script created"

# Create monitoring script
cat > scripts/monitor.sh << 'EOF'
#!/bin/bash

# Real-time monitoring script
echo "🔍 AI Monitoring System Status"
echo "=============================="

# Check service status
echo "🐳 Container Status:"
docker-compose ps

echo ""
echo "📊 Current Metrics:"

# Get current metrics from AI service
if curl -s http://localhost:8000/metrics > /dev/null; then
    echo "AI Service: ✅ Running"
    
    # Parse some key metrics
    FAILURE_PROB=$(curl -s http://localhost:8000/metrics | grep "predicted_system_failure_probability" | grep -v "#" | awk '{print $2}' | head -1)
    HEALTH_SCORE=$(curl -s http://localhost:8000/metrics | grep "system_health_score" | grep -v "#" | awk '{print $2}' | head -1)
    
    echo "Failure Probability: ${FAILURE_PROB:-N/A}"
    echo "Health Score: ${HEALTH_SCORE:-N/A}/100"
else
    echo "AI Service: ❌ Not responding"
fi

echo ""
echo "💾 Disk Usage:"
df -h /

echo ""
echo "🧠 Memory Usage:"
free -h

echo ""
echo "⚡ CPU Load:"
uptime

echo ""
echo "📈 Recent Logs (last 10 lines):"
docker-compose logs --tail=10 ai-monitoring-service
EOF

chmod +x scripts/monitor.sh
print_status "Monitoring script created"

# Final setup
echo -e "${BLUE}🏁 Final setup steps...${NC}"

# Build the AI service
docker-compose build

print_status "Docker images built"

# Test run (don't start permanently)
echo -e "${BLUE}🧪 Testing configuration...${NC}"

if docker-compose up -d --no-deps ai-monitoring-service; then
    sleep 10
    if curl -s http://localhost:8000/metrics > /dev/null; then
        print_status "AI service test successful"
    else
        print_warning "AI service test failed - check logs with: docker-compose logs ai-monitoring-service"
    fi
    docker-compose down
fi

# Create README
cat > README.md <<EOF
EOF
# AI Monitoring System
echo "🤖 Starting AI Monitoring Service..."
## Quick Start

```bash
# Start the system
./scripts/start.sh

# Monitor status
./scripts/monitor.sh

# View logs
docker-compose logs -f

# Stop system
docker-compose down
```

## Features

- ✅ Real-time monitoring via Prometheus (${PROMETHEUS_URL})
- 🤖 AI-powered anomaly detection
- 📈 System failure prediction
- 🎯 Health scoring algorithm
- 📊 Integration with Grafana (${GRAFANA_URL})
- 🔔 Intelligent alerting

## Architecture

- **AI Service**: Python-based ML service
- **Database**: PostgreSQL for historical data
- **Cache**: Redis for performance
- **Monitoring**: Prometheus metrics export
- **Visualization**: Grafana dashboards
