GOFILES := $(shell find . -path './vendor' -prune -o -path './.git' -prune -o -name '*.go' -print)
# The newest x/tools that still builds on the go directive in go.mod.
GOIMPORTS_VERSION := v0.42.0
GOLANGCI_VERSION := v2.12.2

.PHONY: build format format-check lint golangci-lint test cover ci clean install-format-tools check-format-tools check-lint-tools

build:
	go build ./...

format: check-format-tools
	@gofmt -w $(GOFILES)
	@goimports -w $(GOFILES)

format-check: check-format-tools
	@files="$$(gofmt -l $(GOFILES))"; \
	if [ -n "$$files" ]; then \
		echo "Go files are not formatted. Run: make format"; \
		echo "$$files"; \
		exit 1; \
	fi
	@files="$$(goimports -l $(GOFILES))"; \
	if [ -n "$$files" ]; then \
		echo "Go imports are not formatted. Run: make format"; \
		echo "$$files"; \
		exit 1; \
	fi

install-format-tools:
	go install golang.org/x/tools/cmd/goimports@$(GOIMPORTS_VERSION)

check-format-tools:
	@command -v goimports >/dev/null || go install golang.org/x/tools/cmd/goimports@$(GOIMPORTS_VERSION)

check-lint-tools:
	@command -v golangci-lint >/dev/null || \
		go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_VERSION)

lint:
	go vet ./...
	@$(MAKE) format-check

# The linters in .golangci.yml — what CI's own lint job runs. Separate from
# lint so that go vet, which needs no download, still runs on a machine that
# cannot fetch the binary.
golangci-lint: check-lint-tools
	# verify first: CI's action refuses a config the schema rejects before it
	# lints anything, and `run` alone does not check that.
	golangci-lint config verify
	golangci-lint run ./...

test:
	go test -race -timeout 120s ./...

cover:
	go test -race -covermode=atomic -timeout 120s \
		-coverpkg=./pkg/... -coverprofile=coverage.out \
		./...

ci: format-check build lint golangci-lint test

clean:
	rm -f coverage.out
