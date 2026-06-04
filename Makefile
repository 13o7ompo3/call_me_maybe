CACHE_DIR = goinfre/
SRC_DIR = src

all: install run

add:
	@UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" uv add pydantic numpy ./llm_sdk accelerate
	@UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" uv add --dev flake8 mypy

install:
	@UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" uv sync

run:
	@HF_HOME="$(HOME)/$(CACHE_DIR)" uv run python3 -m $(SRC_DIR)

debug:
	@HF_HOME="$(HOME)/$(CACHE_DIR)" uv run python3 -m pdb -m $(SRC_DIR)

clean:
	@rm -rf pycache
	@rm -rf .mypy_cache

lint:
	@uv run flake8 $(SRC_DIR)/$(wildcard .py)
	@uv run mypy $(SRC_DIR)/$(wildcard.py) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	@uv run flake8 $(SRC_DIR)/$(wildcard .py)
	@uv run mypy $(SRC_DIR)/$(wildcard.py) --strict

test:
	@uv run python3 -m pytest