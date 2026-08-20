GOFILES := $(shell find . -path './vendor' -prune -o -path './.git' -prune -o -name '*.go' -print)
GOIMPORTS_VERSION := v0.43.0

.PHONY: build format format-check lint test cover ci clean install-format-tools check-format-tools

build: format
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

lint:
	go vet ./...
	@$(MAKE) format-check

test:
	go test ./...

cover:
	go test -race -covermode=atomic -timeout 120s \
		-coverpkg=./pkg/... -coverprofile=coverage.out \
		./pkg/...

ci: format-check build lint
	$(MAKE) cover

clean:
	rm -f coverage.out
