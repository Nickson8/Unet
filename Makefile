# Load variables from .env
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# --- Magic for positional arguments ---
# Now we only expect ONE argument (the path)
POS_COMMANDS := upload-file upload-dir download-outputs

ifeq ($(filter $(firstword $(MAKECMDGOALS)),$(POS_COMMANDS)),$(firstword $(MAKECMDGOALS)))
  PATH_ARG := $(word 2,$(MAKECMDGOALS))
  $(eval $(PATH_ARG):;@:)
endif

# Settings
KEY         := ~/.ssh/id_ed25519
REMOTE_PATH := /workspace/Unet/
USER        := root

.PHONY: $(POS_COMMANDS)

# --- Recipes ---

upload-file:
	@echo "🚀 Uploading file $(PATH_ARG) to $(SERVER_IP)..."
	scp -P $(SERVER_PORT) -i $(KEY) $(PATH_ARG) $(USER)@$(SERVER_IP):$(REMOTE_PATH)

upload-dir:
	@echo "📂 Uploading directory $(PATH_ARG) to $(SERVER_IP)..."
	scp -P $(SERVER_PORT) -i $(KEY) -r $(PATH_ARG) $(USER)@$(SERVER_IP):$(REMOTE_PATH)

download-outputs:
	@echo "📡 Downloading outputs from $(SERVER_IP) to $(PATH_ARG)..."
	scp -P $(SERVER_PORT) -i $(KEY) -r $(USER)@$(SERVER_IP):$(REMOTE_PATH)/outputs $(PATH_ARG)