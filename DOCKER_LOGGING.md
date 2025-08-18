# 📊 Comprehensive Docker Logging System

## Overview
The AI Data Agent now includes a comprehensive logging system that provides clean API responses while capturing all internal processing details in Docker logs for debugging and monitoring.

## Key Features

### ✨ Clean API Responses
- API endpoints return only the final answer/result
- No internal processing details exposed to users
- Clean format like: `[1, "Titanic", 0.485782, "data:image/png;base64,...]`

### 🔍 Comprehensive Docker Logging
- **Request Tracking**: Each request gets a unique UUID for tracing
- **Workflow Details**: Complete workflow execution logging
- **Code Generation**: All code attempts and retry history
- **Error Analysis**: Full tracebacks and error context
- **Performance Monitoring**: Execution timing and resource usage

## Log Levels and Patterns

### 🚀 REQUEST START
```
🚀 REQUEST START [ID: uuid-here]
📝 Questions: User's question here...
📁 Files provided: 2 files
   - data.csv
   - config.json
```

### 🎯 Workflow Classification
```
🎯 Workflow classification: data_analysis (confidence: 0.85)
💭 Classification reasoning: CSV files detected with data analysis request
```

### ⚙️ Workflow Execution
```
⚙️ Starting workflow execution: data_analysis
📊 Executing workflow: DataAnalysisWorkflow
📊 Manifest data: {...}
```

### 🔄 Code Generation & Retries
```
🔄 Code generation attempts: 3
🔄 Retry attempts: 2
🔄 Retry History (2):
   Retry 1: NameError - variable not defined - Fixed with OpenAI
   Retry 2: TypeError - wrong data type - Fixed with context adjustment
```

### 📊 Complete Workflow Details (DEBUG level)
```
📊 COMPLETE WORKFLOW DETAILS [ID: uuid-here]
📊 Raw workflow result keys: ['result', 'success', 'code_attempts', 'retry_history']
🔄 Code Generation Attempts (3):
   Attempt 1: import pandas as pd...
   Attempt 2: import pandas as pd; df = pd.read_csv...
   Attempt 3: # Fixed version with error handling...
📤 Final extracted answer type: <class 'list'>
📤 Final answer: [1, "Titanic", 0.485782, "data:image/png;base64,iVBOR..."]
```

### ✅ Successful Completion
```
✅ Workflow 'data_analysis' completed successfully [ID: uuid-here]
✅ REQUEST COMPLETE [ID: uuid-here] - Clean response returned
```

### ❌ Error Handling
```
❌ Workflow 'data_analysis' failed [ID: uuid-here]: Database connection error
🔍 Workflow traceback:
Traceback (most recent call last):
  File "workflow.py", line 45, in execute
    ...
❌ REQUEST FAILED [ID: uuid-here]: Database connection error
🔍 Full traceback: ...
```

## Docker Logging Commands

### View Real-time Logs
```bash
docker-compose logs -f ai-data-agent
```

### View Logs with Timestamps
```bash
docker-compose logs -t ai-data-agent
```

### View Last 100 Lines
```bash
docker-compose logs --tail=100 ai-data-agent
```

### Filter by Log Level
```bash
# View only errors
docker-compose logs ai-data-agent | grep ERROR

# View debug details
docker-compose logs ai-data-agent | grep DEBUG

# View workflow execution
docker-compose logs ai-data-agent | grep "⚙️\|📊"

# View request tracking
docker-compose logs ai-data-agent | grep "🚀\|✅\|❌"
```

### Export Logs
```bash
# Export all logs to file
docker-compose logs ai-data-agent > ai_agent_logs.txt

# Export with timestamps
docker-compose logs -t ai-data-agent > ai_agent_logs_timestamped.txt

# Export only today's logs (if using log rotation)
docker logs --since="$(date -d 'today' '+%Y-%m-%d')" ai-data-agent > today_logs.txt
```

### Search Specific Request
```bash
# Find all logs for a specific request ID
docker-compose logs ai-data-agent | grep "uuid-here"

# Find failed requests
docker-compose logs ai-data-agent | grep "REQUEST FAILED"

# Find retry attempts
docker-compose logs ai-data-agent | grep "🔄.*Retry"
```

## Log Configuration

### Environment Variables
```env
LOG_LEVEL=DEBUG          # Set to INFO for production, DEBUG for detailed logging
PYTHONUNBUFFERED=1       # Ensures immediate log output
```

### Docker Compose Configuration
```yaml
services:
  ai-data-agent:
    environment:
      - PYTHONUNBUFFERED=1
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
    labels:
      - "logging.level=debug"
      - "service.name=ai-data-agent"
```

## Monitoring and Alerting

### Key Metrics to Monitor
- Request completion rate
- Average processing time
- Error rates by workflow type
- Retry attempt frequency
- Resource usage patterns

### Log Analysis Patterns
```bash
# Count requests per hour
docker-compose logs ai-data-agent | grep "REQUEST START" | awk '{print $1" "$2}' | cut -d: -f1-2 | sort | uniq -c

# Find most common errors
docker-compose logs ai-data-agent | grep "ERROR" | sort | uniq -c | sort -nr

# Track workflow performance
docker-compose logs ai-data-agent | grep "Workflow.*completed" | awk '{print $NF}' | sort | uniq -c
```

## Troubleshooting Guide

### Common Issues

1. **No Logs Visible**
   ```bash
   # Check container status
   docker-compose ps
   
   # Check if container is running
   docker-compose logs ai-data-agent --tail=50
   ```

2. **Missing DEBUG Logs**
   ```bash
   # Set environment variable
   docker-compose exec ai-data-agent env | grep LOG_LEVEL
   
   # Restart with DEBUG level
   LOG_LEVEL=DEBUG docker-compose up -d ai-data-agent
   ```

3. **Request Not Completing**
   ```bash
   # Search for request ID
   docker-compose logs ai-data-agent | grep "REQUEST START" | tail -10
   
   # Check for corresponding completion
   docker-compose logs ai-data-agent | grep "REQUEST COMPLETE\|REQUEST FAILED"
   ```

## Best Practices

### For Development
- Use `LOG_LEVEL=DEBUG` for detailed internal processing
- Monitor logs in real-time during testing: `docker-compose logs -f ai-data-agent`
- Export logs for analysis: `docker-compose logs ai-data-agent > debug_session.log`

### For Production
- Use `LOG_LEVEL=INFO` to reduce log volume
- Set up log rotation: max-size: "100m", max-file: "10"
- Monitor for ERROR and WARNING patterns
- Set up alerts for failed requests

### For Debugging
- Search by request ID for complete request lifecycle
- Look for `🔄 Retry` patterns to identify problematic code
- Check `📊 COMPLETE WORKFLOW` logs for detailed execution analysis
- Monitor `❌ REQUEST FAILED` for system issues

## Integration with Monitoring Tools

### ELK Stack (Elasticsearch, Logstash, Kibana)
```yaml
# Add to docker-compose.yml for log shipping
logging:
  driver: "json-file"
  options:
    labels: "service.name,logging.level"
```

### Grafana + Loki
```yaml
# Configure Loki log driver
logging:
  driver: loki
  options:
    loki-url: "http://loki:3100/loki/api/v1/push"
```

This comprehensive logging system ensures you get clean, user-friendly API responses while maintaining complete visibility into all internal processing for debugging and monitoring purposes.
