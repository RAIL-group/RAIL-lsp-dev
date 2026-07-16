import os
import httpx

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434/v1", model="gemma:4b"):
        # Map base_url to the native Ollama chat API
        url = base_url.strip()
        if url.endswith("/v1"):
            self.chat_url = url[:-3] + "/api/chat"
        elif url.endswith("/v1/"):
            self.chat_url = url[:-4] + "/api/chat"
        elif "/api/chat" not in url:
            self.chat_url = url.rstrip("/") + "/api/chat"
        else:
            self.chat_url = url

        self.tags_url = self.chat_url.replace("/api/chat", "/api/tags")
        self.model = model
        self._model_resolved = False

    def query(self, prompt, response_format=None, system_prompt=None):
        if not self._model_resolved:
            self._resolve_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Base request payload for Ollama native API
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": 128000,
                "temperature": 1.0
            }
        }

        # Ollama supports format="json" natively for JSON mode
        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"

        try:
            response = httpx.post(self.chat_url, json=payload, timeout=120.0)
            response.raise_for_status()
            res_json = response.json()
            return res_json["message"]["content"].strip()
        except Exception as e:
            raise ConnectionError(
                f"Failed to query local LLM server at {self.chat_url} (model: {self.model}): {e}"
            )

    def _resolve_model(self):
        try:
            response = httpx.get(self.tags_url, timeout=10.0)
            response.raise_for_status()
            models_data = response.json().get("models", [])
            available_models = [m["name"] for m in models_data]

            if self.model not in available_models:
                # Try matching after normalization
                def normalize(name):
                    return name.lower().replace("-", "").replace(":", "").replace("_", "")

                target = normalize(self.model)
                matched = False
                for m in available_models:
                    if normalize(m) == target:
                        print(f"Resolving requested model '{self.model}' to available Ollama model '{m}'")
                        self.model = m
                        matched = True
                        break

                if not matched:
                    # Fallback to the first available model containing 'gemma'
                    gemma_models = [m for m in available_models if "gemma" in m.lower()]
                    if gemma_models:
                        print(f"Model '{self.model}' not found. Falling back to '{gemma_models[0]}'")
                        self.model = gemma_models[0]
                    elif available_models:
                        print(f"Model '{self.model}' not found. Falling back to '{available_models[0]}'")
                        self.model = available_models[0]
            self._model_resolved = True
        except Exception as e:
            # Ignore listing exceptions and try with whatever model was requested
            print(f"Warning: Failed to list/resolve models from local server: {e}")
            self._model_resolved = True


