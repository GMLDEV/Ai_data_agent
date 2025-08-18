# 🐳 Docker/Linux Deployment Configuration

## Environment Compatibility Updates

The sandbox executor has been updated for optimal Linux/Docker container performance:

### 🔧 Key Changes for Linux/Docker

#### 1. **Path Separator Handling**
```python
# Automatic detection based on platform
path_separator = ":" if platform.system() != "Windows" else ";"

# PYTHONPATH construction
env['PYTHONPATH'] = temp_dir + (path_separator + current_pythonpath if current_pythonpath else '')
```

#### 2. **Local Package Installation**
```bash
# Docker-optimized pip command
pip install -r requirements.txt \
  --target /tmp/sandbox_dir \
  --no-cache-dir \
  --disable-pip-version-check \
  --upgrade \
  --force-reinstall
```

#### 3. **Environment Isolation**
```python
# Consistent environment across install/verify/execute
env = {
    'PYTHONPATH': '/tmp/sandbox_dir:/original/pythonpath',
    'HOME': '/tmp/sandbox_dir',
    # Other environment variables preserved
}
```

### 🐳 Docker Deployment

#### Recommended Dockerfile
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install base dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

#### Docker Compose Configuration
```yaml
version: '3.8'
services:
  ai-data-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - PYTHONPATH=/app
    volumes:
      - ./data:/app/data
    tmpfs:
      - /tmp  # For sandbox isolation
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### 🚀 Cloud Droplet Deployment

#### System Requirements
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-pip python3.12-venv git

# Create virtual environment
python3.12 -m venv ai_data_agent_venv
source ai_data_agent_venv/bin/activate

# Install application
git clone <repo>
cd ai_data_agent
pip install -r requirements.txt

# Run with systemd service
sudo systemctl enable ai-data-agent
sudo systemctl start ai-data-agent
```

#### Systemd Service File (`/etc/systemd/system/ai-data-agent.service`)
```ini
[Unit]
Description=AI Data Agent
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/ai_data_agent
Environment=PYTHONPATH=/opt/ai_data_agent
Environment=LOG_LEVEL=INFO
ExecStart=/opt/ai_data_agent/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 🔍 Linux-Specific Testing

#### Environment Variable Verification
```python
# The system automatically detects Linux and uses:
- Path separator: ':'
- Package installation: --target with local directory
- PYTHONPATH: Sandbox directory first, then system paths
- HOME: Set to sandbox directory for isolation
```

#### Package Installation Flow
```
1. Create temporary directory: /tmp/ai_data_agent_XXXXXX
2. Install packages: pip install --target /tmp/ai_data_agent_XXXXXX
3. Set PYTHONPATH: /tmp/ai_data_agent_XXXXXX:/usr/local/lib/python3.12/site-packages
4. Execute code with consistent environment
5. Verify package accessibility
6. Clean up temporary directory
```

### 🚨 Troubleshooting Linux/Docker Issues

#### Common Issues

1. **Permission Errors**
   ```bash
   # Ensure proper permissions in Docker
   USER 1000:1000
   ```

2. **Package Not Found**
   ```bash
   # Check PYTHONPATH in logs
   grep "PYTHONPATH" docker_logs.txt
   ```

3. **Import Errors**
   ```python
   # Verify environment consistency
   logger.debug(f"Platform: {platform.system()}")
   logger.debug(f"Python executable: {sys.executable}")
   logger.debug(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
   ```

#### Monitoring Commands
```bash
# Check container logs
docker logs ai_data_agent | grep "Environment mismatch"

# Monitor package installations
docker logs ai_data_agent | grep "Successfully installed"

# Check fallback usage
docker logs ai_data_agent | grep "fallback_triggered"
```

### 📊 Performance Optimization

#### Linux-Specific Optimizations
- `--no-cache-dir`: Prevents Docker layer bloat
- `--target`: Local installation for isolation
- `--force-reinstall`: Ensures clean package state
- Temporary filesystem: `/tmp` for sandbox isolation

#### Resource Limits
```yaml
# Docker resource limits
deploy:
  resources:
    limits:
      memory: 2G      # Sufficient for data analysis
      cpus: '1.0'     # Single CPU for sandbox
    reservations:
      memory: 512M    # Minimum memory
```

### ✅ Verification Checklist

Before deploying to Linux/Docker:
- [ ] Path separators use ':' on Linux
- [ ] Package installation uses `--target`
- [ ] Environment variables consistent across install/execute
- [ ] PYTHONPATH includes sandbox directory first
- [ ] Temporary directories cleaned up properly
- [ ] Logging includes platform information
- [ ] Error handling works in container environment

This configuration ensures the AI Data Agent works reliably in Linux containers and cloud droplets while maintaining the same functionality as the Windows development environment.
