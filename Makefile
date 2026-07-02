# Makefile for the AI guidelines pipeline.
#
# Every stage is incremental: the scripts skip anything already produced and
# only process missing items. Pass FORCE=1 to reprocess everything, e.g.
#
#     make all              # full pipeline, non-forced (only new work)
#     make extract          # just the extraction stage, non-forced
#     make all FORCE=1      # re-run every stage from scratch
#
# Individual stages can be run on a single slug, e.g.
#
#     make download SLUG=universitat-mannheim

UV        ?= uv run
SCRIPTS   := scripts
CLEAN_DIR := texts_clean
EXPLORER  := web/explorer_data.json

# When FORCE=1, append --force to the stages that support it.
ifeq ($(FORCE),1)
FORCE_FLAG := --force
else
FORCE_FLAG :=
endif

# When SLUG is set, operate on that one slug; otherwise operate on everything.
ifdef SLUG
SEL       := --slug $(SLUG)
else
SEL       := --all
endif

.PHONY: all validate download extract clean code toc references references-index index explorer help

## all: run the whole pipeline in order (non-forced by default)
all: validate download extract clean code toc references references-index index explorer

## validate: check guidelines.yaml metadata
validate:
	$(UV) $(SCRIPTS)/validate.py

## download: fetch source documents (skips already-downloaded)
download:
	$(UV) $(SCRIPTS)/download.py $(SEL) $(FORCE_FLAG)

## extract: extract raw text (skips already-extracted)
extract:
	$(UV) $(SCRIPTS)/extract_text.py $(SEL) $(FORCE_FLAG)

## clean: LLM-clean extracted text (skips already-cleaned)
clean:
	$(UV) $(SCRIPTS)/clean_texts.py $(SEL) $(FORCE_FLAG)

## code: run B-code segmentation over cleaned texts (skips already-coded)
code:
	$(UV) $(SCRIPTS)/run_coding.py --dir $(CLEAN_DIR) $(FORCE_FLAG)

## toc: extract tables of contents from cleaned texts (skips existing)
toc:
	$(UV) $(SCRIPTS)/run_toc.py --dir $(CLEAN_DIR) $(FORCE_FLAG)

## references: extract referenced external documents from cleaned texts (skips existing)
references:
	$(UV) $(SCRIPTS)/run_references.py --dir $(CLEAN_DIR) $(FORCE_FLAG)

## references-index: merge per-document references into one deduplicated list
references-index:
	$(UV) $(SCRIPTS)/aggregate_references.py

## index: rebuild index.json from metadata + file presence
index:
	$(UV) $(SCRIPTS)/build_index.py

## explorer: rebuild the explorer dataset for web/
explorer:
	$(UV) $(SCRIPTS)/build_explorer_data.py --output $(EXPLORER)

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'
