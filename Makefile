SHELL := /bin/sh

PROJECT_VERSION = $(shell uv version --short)
VERSION ?= $(PROJECT_VERSION)
TAG = v$(VERSION)
RELEASE_DIST = dist/release-$(VERSION)

.DEFAULT_GOAL := help

.PHONY: help check build release-check release

help:
	@printf '%s\n' \
		'make check                    Run tests and lint' \
		'make build                    Build and validate distributions' \
		'make release VERSION=0.1.0    Tag and open a GitHub release'

check:
	uv run --no-active --quiet pytest
	uv run --no-active --quiet ruff check src tests

build: check
	uv build --out-dir "$(RELEASE_DIST)"
	uvx twine check "$(RELEASE_DIST)"/*

release-check:
	@test "$(VERSION)" = "$(PROJECT_VERSION)" || { \
		echo "VERSION=$(VERSION) does not match pyproject.toml ($(PROJECT_VERSION))."; \
		echo "Run: uv version $(VERSION)"; \
		exit 1; \
	}
	@test -z "$$(git status --porcelain)" || { \
		echo "The worktree is not clean. Commit the release changes first."; \
		exit 1; \
	}
	@command -v gh >/dev/null 2>&1 || { \
		echo "The GitHub CLI (gh) is required to create the release."; \
		exit 1; \
	}
	@gh auth status >/dev/null 2>&1 || { \
		echo "The GitHub CLI is not authenticated. Run: gh auth login"; \
		exit 1; \
	}
	@if git rev-parse --quiet --verify "refs/tags/$(TAG)" >/dev/null; then \
		echo "Tag $(TAG) already exists locally."; \
		exit 1; \
	fi
	@if git ls-remote --exit-code --tags origin "refs/tags/$(TAG)" >/dev/null 2>&1; then \
		echo "Tag $(TAG) already exists on origin."; \
		exit 1; \
	fi
	@$(MAKE) check
	uv build --out-dir "$(RELEASE_DIST)"
	uvx twine check "$(RELEASE_DIST)"/*

release: release-check
	git tag -a "$(TAG)" -m "jev-align $(VERSION)"
	git push origin "$(TAG)"
	gh release create "$(TAG)" --verify-tag --generate-notes --title "jev-align $(VERSION)"
