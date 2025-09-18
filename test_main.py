from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from main import app, load_model, model, tokenizer

client = TestClient(app)


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    mock_model = Mock()
    mock_model.device = "cuda:0"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 151668, 6, 7, 8]])
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing"""
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "Test template"
    mock_tokenizer.return_value = Mock()
    mock_tokenizer.return_value.to.return_value = Mock()
    mock_tokenizer.return_value.input_ids = [[1, 2, 3, 4]]
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.side_effect = ["thinking content", "response content"]
    return mock_tokenizer


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_success(self):
        """Test successful health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_endpoint(self):
        """Test root endpoint returns correct message"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Qwen Thinking Model API"
        assert data["docs"] == "/docs"


class TestChatEndpoint:
    """Test chat endpoint"""

    @patch("main.model")
    @patch("main.tokenizer")
    def test_chat_success(self, mock_tokenizer_patch, mock_model_patch):
        """Test successful chat generation"""
        # Setup mocks
        mock_model_patch.device = "cuda:0"
        mock_model_patch.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 151668, 6, 7, 8]])

        mock_tokenizer_patch.apply_chat_template.return_value = "Test template"
        mock_tokenizer_patch.return_value = Mock()
        mock_tokenizer_patch.return_value.to.return_value = Mock()
        mock_tokenizer_patch.return_value.input_ids = [[1, 2, 3, 4]]
        mock_tokenizer_patch.eos_token_id = 0
        mock_tokenizer_patch.decode.side_effect = ["thinking content", "response content"]

        # Make request
        response = client.post("/chat", json={"prompt": "Test prompt", "max_new_tokens": 100})

        assert response.status_code == 200
        data = response.json()
        assert "thinking_content" in data
        assert "content" in data
        assert "prompt" in data
        assert data["prompt"] == "Test prompt"

    def test_chat_model_not_loaded(self):
        """Test chat when model is not loaded"""
        with patch("main.model", None), patch("main.tokenizer", None):
            response = client.post("/chat", json={"prompt": "Test prompt"})
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

    @patch("main.model")
    @patch("main.tokenizer")
    def test_chat_with_custom_parameters(self, mock_tokenizer_patch, mock_model_patch):
        """Test chat with custom generation parameters"""
        # Setup mocks
        mock_model_patch.device = "cuda:0"
        mock_model_patch.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 151668, 6, 7, 8]])

        mock_tokenizer_patch.apply_chat_template.return_value = "Test template"
        mock_tokenizer_patch.return_value = Mock()
        mock_tokenizer_patch.return_value.to.return_value = Mock()
        mock_tokenizer_patch.return_value.input_ids = [[1, 2, 3, 4]]
        mock_tokenizer_patch.eos_token_id = 0
        mock_tokenizer_patch.decode.side_effect = ["thinking content", "response content"]

        response = client.post(
            "/chat",
            json={
                "prompt": "Test prompt",
                "max_new_tokens": 256,
                "temperature": 0.8,
                "top_p": 0.95,
            },
        )

        assert response.status_code == 200

        # Verify model.generate was called with correct parameters
        mock_model_patch.generate.assert_called_once()
        call_kwargs = mock_model_patch.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 256
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95

    @patch("main.model")
    @patch("main.tokenizer")
    def test_chat_no_thinking_tag(self, mock_tokenizer_patch, mock_model_patch):
        """Test chat when no thinking tag is found"""
        # Setup mocks - no 151668 token in output
        mock_model_patch.device = "cuda:0"
        mock_model_patch.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        mock_tokenizer_patch.apply_chat_template.return_value = "Test template"
        mock_tokenizer_patch.return_value = Mock()
        mock_tokenizer_patch.return_value.to.return_value = Mock()
        mock_tokenizer_patch.return_value.input_ids = [[1, 2, 3, 4]]
        mock_tokenizer_patch.eos_token_id = 0
        mock_tokenizer_patch.decode.side_effect = ["", "full response content"]

        response = client.post("/chat", json={"prompt": "Test prompt"})

        assert response.status_code == 200
        data = response.json()
        assert data["thinking_content"] == ""
        assert data["content"] == "full response content"

    @patch("main.model")
    @patch("main.tokenizer")
    def test_chat_generation_error(self, mock_tokenizer_patch, mock_model_patch):
        """Test chat when generation fails"""
        mock_model_patch.generate.side_effect = Exception("Generation failed")
        mock_tokenizer_patch.apply_chat_template.return_value = "Test template"
        mock_tokenizer_patch.return_value = Mock()
        mock_tokenizer_patch.return_value.to.return_value = Mock()
        mock_tokenizer_patch.return_value.input_ids = [[1, 2, 3, 4]]

        response = client.post("/chat", json={"prompt": "Test prompt"})

        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]

    def test_chat_invalid_request(self):
        """Test chat with invalid request body"""
        response = client.post("/chat", json={"invalid_field": "value"})
        assert response.status_code == 422  # Validation error


class TestModelLoading:
    """Test model loading functionality"""

    @patch("main.AutoModelForCausalLM.from_pretrained")
    @patch("main.AutoTokenizer.from_pretrained")
    def test_load_model_success(self, mock_tokenizer_load, mock_model_load):
        """Test successful model loading"""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_load.return_value = mock_model
        mock_tokenizer_load.return_value = mock_tokenizer

        # Clear global variables
        import main

        main.model = None
        main.tokenizer = None

        load_model()

        assert main.model == mock_model
        assert main.tokenizer == mock_tokenizer
        mock_tokenizer_load.assert_called_once_with("Qwen/Qwen3-4B-Thinking-2507-FP8")
        mock_model_load.assert_called_once()

    @patch("main.AutoModelForCausalLM.from_pretrained")
    @patch("main.AutoTokenizer.from_pretrained")
    def test_load_model_already_loaded(self, mock_tokenizer_load, mock_model_load):
        """Test that model loading is skipped when already loaded"""
        import main

        main.model = Mock()
        main.tokenizer = Mock()

        load_model()

        # Should not call loading functions again
        mock_tokenizer_load.assert_not_called()
        mock_model_load.assert_not_called()

    @patch("main.AutoTokenizer.from_pretrained")
    def test_load_model_failure(self, mock_tokenizer_load):
        """Test model loading failure"""
        mock_tokenizer_load.side_effect = Exception("Loading failed")

        import main

        main.model = None
        main.tokenizer = None

        with pytest.raises(Exception) as exc_info:
            load_model()

        assert "Loading failed" in str(exc_info.value)


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
