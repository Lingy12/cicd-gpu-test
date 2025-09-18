from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from main import app, load_model, model, tokenizer

client = TestClient(app)

@pytest.mark.integration
class TestIntegration:
    """Integration tests - require running server"""

    def test_server_health_check(self):
        """Test server health check endpoint against running server"""
        import time

        import requests

        # Give server a moment to be fully ready
        time.sleep(1)

        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "model_loaded" in data
            assert "device" in data
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Failed to connect to running server: {e}")

    def test_server_root_endpoint(self):
        """Test server root endpoint against running server"""
        import time

        import requests

        # Give server a moment to be fully ready
        time.sleep(1)

        try:
            response = requests.get("http://localhost:8000/", timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Qwen Thinking Model API"
            assert data["docs"] == "/docs"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Failed to connect to running server: {e}")

    def test_chat_endpoint_with_sample_content(self):
        """Test chat endpoint with sample content against running server"""
        import time

        import requests

        # Give server a moment to be fully ready
        time.sleep(2)

        # Sample content to send
        sample_prompts = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "Write a short poem about nature",
        ]

        for prompt in sample_prompts:
            try:
                payload = {
                    "prompt": prompt,
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }

                response = requests.post("http://localhost:8000/chat", json=payload, timeout=30)

                # Check response structure
                assert response.status_code == 200
                data = response.json()

                # Verify required fields are present
                assert "thinking_content" in data
                assert "content" in data
                assert "prompt" in data
                assert data["prompt"] == prompt

                # Verify content is not empty (indicates model is working)
                assert isinstance(data["thinking_content"], str)
                assert isinstance(data["content"], str)
                assert isinstance(data["prompt"], str)

                print(f"✓ Successfully tested prompt: '{prompt[:50]}...'")
                print(f"  Response length: {len(data['content'])} chars")

            except requests.exceptions.RequestException as e:
                pytest.fail(f"Failed to connect to running server for prompt '{prompt}': {e}")
            except Exception as e:
                pytest.fail(f"Unexpected error for prompt '{prompt}': {e}")

    @pytest.mark.skip(reason="Requires actual model download")
    def test_full_pipeline_with_model(self):
        """Test full pipeline with actual model (skipped by default)"""
        # This test would require actual model loading
        # and would be slow, so it's skipped by default
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
