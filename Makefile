# Makefile for Directus Template Extraction

# Variables
# Note: Removed unnecessary quotes from variable assignment.
# Quotes are kept in the command itself where needed.
TEMPLATE_NAME := Aqueduct AI Gateway
TEMPLATE_LOCATION := migrate/template
DIRECTUS_URL?=http://localhost:8055
DIRECTUS_TOKEN=

# Declare phony targets (targets that don't represent files)
.PHONY: extract help

# Default target (optional - runs 'make help' if you just type 'make')
default: help

# Target to extract the Directus template
# Renamed from .extract to extract
extract:
	@echo "Attempting to extract template '$(TEMPLATE_NAME)' from $(DIRECTUS_URL)...";
	npx directus-template-cli@latest extract \
		-p \
		--templateName="$(TEMPLATE_NAME)" \
		--templateLocation="$(TEMPLATE_LOCATION)" \
		--directusToken="$(DIRECTUS_TOKEN)" \
		--directusUrl="$(DIRECTUS_URL)"
	# Replace whatever content was in the template with an empty list
	find ./migrate/template/src/content -type f -exec sh -c 'echo "[]" > "$$1"' sh {} \;
	echo "[]" > ./migrate/template/src/users.json
	# Remove Administrator role from roles.json as it will be added again in the Directus initialization
	jq 'map(select(.name != "Administrator"))' ./migrate/template/src/roles.json > temp.json && mv temp.json ./migrate/template/src/roles.json


# Target to display help information
# Renamed from .help to help
help:
	@echo "Usage: make <target> [VARIABLE=value]"
	@echo ""
	@echo "Targets:"
	@echo "  extract    Extract the Directus template. Requires DIRECTUS_TOKEN."
	@echo "             Prompts for Directus Admin URL if not provided."
	@echo "  help       Display this help message."
	@echo ""
	@echo "Configuration Variables (can be overridden on the command line):"
	@echo "  TEMPLATE_NAME     : $(TEMPLATE_NAME) (Name of the template to extract)"
	@echo "  TEMPLATE_LOCATION : $(TEMPLATE_LOCATION) (Path to save the extracted template)"
	@echo "  DIRECTUS_URL      : $(DIRECTUS_URL) (Directus instance URL)"
	@echo "  DIRECTUS_TOKEN    : $(DIRECTUS_TOKEN) (Directus Admin Token - REQUIRED)"
	@echo ""
	@echo "Examples:"
	@echo "  make extract DIRECTUS_TOKEN=my-token"
	@echo "  make extract DIRECTUS_URL=http://my-directus:8055 DIRECTUS_TOKEN=my-token"
	@echo "  make extract TEMPLATE_NAME=\"My Custom Template\" DIRECTUS_TOKEN=my-token"	