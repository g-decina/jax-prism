.PHONY: install-mac install-cuda clean check

# For Mac (Metal/MPS) and standard CPU dev
install-mac:
	@echo "Installing for MacOS (MPS/CPU)..."
	poetry install
	@echo "Setup complete. Do'nt forget to run: source .envrc"

# For Linux/CUDA
install-cuda:
	@echo "Installing for CUDA 12..."
	poetry install
	poetry run pip install --upgrade "jax[cuda12_pip]=0.8.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Quality Control
check:
	poetry run ruff check src/
	poetry run pytest tests/
clean:
	rm -rf .pytest_cache
	rm -rf dist
	find . -type d -name "__pycache__" -exec rm -rf {} +