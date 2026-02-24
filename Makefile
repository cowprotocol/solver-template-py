# CoW Protocol Solver Template - Makefile
# Poetry-based commands for development and testing

.PHONY: help run format test clean install

# Default target
help:
	@echo "CoW Protocol Solver Template - Available Commands:"
	@echo ""
	@echo "  make run      - Start the solver server"
	@echo "  make format   - Format code with black"
	@echo "  make test     - Run all tests"
	@echo "  make install  - Install dependencies with Poetry"
	@echo "  make clean    - Clean up temporary files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make run      # Start server on http://localhost:8080"
	@echo "  make format   # Format all Python code"
	@echo "  make test     # Run test suite"

# Start the solver server
run:
	@echo "🚀 Starting CoW Protocol Solver..."
	@echo "Server will be available at: http://localhost:8080"
	@echo "Press Ctrl+C to stop"
	@echo ""
	poetry run python -m src.infra.cli run

# Format code with black
format:
	@echo "🎨 Formatting code with black..."
	poetry run black src/ --line-length 88
	@echo "✅ Code formatting complete!"

# Run tests
test:
	@echo "🧪 Running test suite..."
	poetry run pytest src/tests/ -v
	@echo "✅ Tests complete!"

# Install dependencies
install:
	@echo "📦 Installing dependencies with Poetry..."
	poetry install
	@echo "✅ Installation complete!"
	@echo "Dependencies are managed by Poetry"

# Clean up temporary files
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	@echo "✅ Cleanup complete!"

# Poetry-specific commands
poetry-install:
	@echo "📦 Installing dependencies with Poetry..."
	poetry install

poetry-update:
	@echo "🔄 Updating dependencies..."
	poetry update

poetry-check:
	@echo "🔍 Checking dependencies..."
	poetry check

poetry-export:
	@echo "📤 Exporting requirements.txt for compatibility..."
	poetry export -f requirements.txt --output requirements.txt --without-hashes