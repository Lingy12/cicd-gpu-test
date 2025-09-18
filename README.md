# Qwen Thinking Model API

A FastAPI-based REST API for the Qwen3-4B-Thinking-2507-FP8 model with GPU support and comprehensive CI/CD pipeline using self-hosted runners.

## Features

- 🚀 FastAPI-based REST API for Qwen thinking model inference
- 🧠 Separates thinking content from response content
- 🐳 Docker containerization with CUDA support
- 🧪 Comprehensive test suite with mocking
- 🔄 CI/CD pipeline with self-hosted GPU runners
- 📊 Health monitoring and logging
- ⚡ GPU acceleration support

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the API and model loading state.

### Chat Completion
```
POST /chat
```
Generate responses using the Qwen thinking model.

**Request Body:**
```json
{
  "prompt": "Your question here",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "thinking_content": "Model's internal thinking process",
  "content": "Final response to the user",
  "prompt": "Original prompt"
}
```

### API Documentation
```
GET /docs
```
Interactive API documentation (Swagger UI).

## Installation

### Local Development

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cicd-gpu-test
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the API:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`.

### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t qwen-api .
```

2. **Run with GPU support:**
```bash
docker run --gpus all -p 8000:8000 qwen-api
```

3. **Run without GPU (CPU only):**
```bash
docker run -p 8000:8000 qwen-api
```

## Testing

### Unit Tests
```bash
pytest test_main.py -v
```

### With Coverage
```bash
pytest test_main.py --cov=main --cov-report=html
```

### Integration Tests (requires model)
```bash
pytest test_main.py -m integration
```

## CI/CD Pipeline

The project includes a comprehensive GitHub Actions workflow that supports self-hosted runners with GPU capabilities.

### Workflow Jobs

1. **Lint and Test (CPU)** - Runs on GitHub-hosted runners
   - Code linting with flake8
   - Code formatting check with black
   - Import sorting check with isort
   - Unit tests with pytest
   - Coverage reporting

2. **GPU Integration Test** - Runs on self-hosted GPU runners
   - GPU availability check
   - Model loading test
   - API health check
   - Integration tests

3. **Build and Deploy** - Runs on self-hosted runners
   - Docker image building
   - Container testing
   - Deployment to staging/production

4. **Cleanup** - Cleans up resources on self-hosted runners

### Setting Up Self-Hosted Runners

1. **Install GitHub Actions Runner:**
   Follow [GitHub's documentation](https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners) to set up a self-hosted runner.

2. **Configure GPU Support:**
   ```bash
   # Install NVIDIA Docker runtime
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   
   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
   ```

3. **Add Runner Labels:**
   Configure your self-hosted runner with labels: `self-hosted`, `gpu`, `cuda`

### Environment Variables

Set these environment variables in your GitHub repository settings:

- `MODEL_CACHE_DIR`: Directory for caching downloaded models
- `HUGGING_FACE_HUB_TOKEN`: (Optional) For private model access

## Configuration

### Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `MODEL_CACHE_DIR`: Model cache directory
- `HF_HOME`: Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Transformers cache directory

### Model Configuration

The API uses the Qwen3-4B-Thinking-2507-FP8 model. You can modify the model name in `main.py`:

```python
model_name = "Qwen/Qwen3-4B-Thinking-2507-FP8"
```

## Performance Considerations

- **GPU Memory**: The model requires significant GPU memory. Ensure your GPU has at least 8GB VRAM.
- **CPU Fallback**: The API automatically falls back to CPU if GPU is not available.
- **Model Caching**: Models are cached locally to avoid repeated downloads.
- **Batch Processing**: Consider implementing batch processing for multiple requests.

## Monitoring

The API includes basic health monitoring:

- Health check endpoint at `/health`
- Logging with configurable levels
- Docker health checks

For production deployment, consider adding:
- Prometheus metrics
- Application performance monitoring
- Error tracking (e.g., Sentry)

## Security

- Non-root user in Docker container
- Input validation with Pydantic
- Environment variable configuration
- Dependencies pinned to specific versions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce `max_new_tokens`
   - Use CPU-only mode
   - Restart the API to clear GPU memory

2. **Model Loading Fails:**
   - Check internet connection
   - Verify model name
   - Check available disk space
   - Ensure Hugging Face access token (if required)

3. **API Startup Slow:**
   - Model download can take time on first run
   - Check network speed
   - Consider pre-downloading models

### Logs

Check application logs for detailed error information:
```bash
docker logs <container-id>
```

## Development

### Code Style

The project uses:
- `black` for code formatting
- `flake8` for linting
- `isort` for import sorting

Run formatting:
```bash
black .
isort .
```

### Adding New Features

1. Implement the feature in `main.py`
2. Add corresponding tests in `test_main.py`
3. Update documentation
4. Ensure CI/CD pipeline passes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure CI/CD passes
6. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions:
- Open a GitHub issue
- Check the troubleshooting section
- Review the CI/CD logs for deployment issues
