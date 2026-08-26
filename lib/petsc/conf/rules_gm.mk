# Review the current branch's changes with an LLM CLI (CLI refers to a command line tool such as claude that runs Claude Code)
# By default this uses Claude Code, override this by exporting PETSC_LLM_CLI
# Use PETSC_LLM_MODEL to select a particular model, for example, PETSC_LLM_MODEL=opus
# For other LLM CLI not listed below provide PETSC_LLM_CLI_OPTS, for example PETSC_LLM_CLI_OPTS=--prompt
PETSC_LLM_CLI ?= claude
ifeq ($(PETSC_LLM_CLI),codex)
    PETSC_LLM_CLI_OPTS ?= exec
endif
ifeq ($(PETSC_LLM_CLI),claude)
    PETSC_LLM_CLI_OPTS ?= --dangerously-skip-permissions
endif
ifeq ($(PETSC_LLM_CLI),gemini)
    PETSC_LLM_CLI_OPTS ?= --approval-mode yolo --prompt-interactive
endif
ifeq ($(PETSC_LLM_CLI),opencode)
    PETSC_LLM_CLI_OPTS ?= --prompt
endif
ifdef PETSC_LLM_MODEL
    PETSC_LLM_MODEL_OPTION := --model $(PETSC_LLM_MODEL)
endif

.PHONY: branch-review
branch-review:
	@command -v $(PETSC_LLM_CLI) >/dev/null 2>&1 || { echo "$(PETSC_LLM_CLI) not installed"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not installed"; exit 1; }
	@git diff --quiet HEAD || { echo "Git repository has uncommitted changes"; exit 1; }
	@$(PETSC_LLM_CLI) $(PETSC_LLM_MODEL_OPTION) $(PETSC_LLM_CLI_OPTS) 'Use the skill review-branch'
