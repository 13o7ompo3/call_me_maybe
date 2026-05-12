PACKAGE_MANAGER = uv
CACHE_DIR = goinfre/
SRC_DIR = src

all: install run

add:
	@UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" $(PACKAGE_MANAGER) add pydantic numpy ./llm_sdk
	@UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" $(PACKAGE_MANAGER) add --dev flake8 mypy

install:
	@UV_CACHE_DIR="$(HOME)/$(CACHE_DIR)" $(PACKAGE_MANAGER) sync

run:
	@HF_HOME="$(HOME)/$(CACHE_DIR)" $(PACKAGE_MANAGER) run python3 -m $(SRC_DIR)

debug:
	@HF_HOME="$(HOME)/$(CACHE_DIR)" $(PACKAGE_MANAGER) run python3 -m pdb -m $(SRC_DIR)

clean:
	@rm -rf pycache
	@rm -rf .mypy_cache

lint:
	@$(PACKAGE_MANAGER) run flake8 $(SRC_DIR)/$(wildcard .py)
	@$(PACKAGE_MANAGER) run mypy $(SRC_DIR)/$(wildcard.py) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	@$(PACKAGE_MANAGER) run flake8 $(SRC_DIR)/$(wildcard .py)
	@$(PACKAGE_MANAGER) run mypy $(SRC_DIR)/$(wildcard.py) --strict

test:
	@$(PACKAGE_MANAGER) run python3 -m pytest