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
