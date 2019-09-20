GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
BINARY_NAME=cmd/nn
BINARY_UNIX=$(BINARY_NAME)_unix

all: test build
build: 
		@echo "  >  Building binary..."
		$(GOBUILD) -o $(BINARY_NAME) -v
test: 
		@echo "  >  Running tests..."
		$(GOTEST) ./...
clean: 
		$(GOCLEAN)
		rm -f $(BINARY_NAME)
		rm -f $(BINARY_UNIX)
run:
		$(GOBUILD) -o $(BINARY_NAME) -v
		./$(BINARY_NAME)
