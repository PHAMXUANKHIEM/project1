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
