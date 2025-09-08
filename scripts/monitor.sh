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
