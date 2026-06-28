#!/bin/bash
# Clone the external repositories this project builds on.
# These are NOT tracked in git (see .gitignore); run this once locally.
#
# Checkpoints are NOT downloaded here — follow each repo's instructions, e.g.:
#   SAM-2:        sam2/checkpoints/sam2.1_hiera_large.pt
set -e

clone() {  # clone <url> <dir>
    if [ -d "$2/.git" ] || [ -d "$2" ]; then
        echo "== $2 already present, skipping"
    else
        echo "== cloning $2"
        git clone --depth 1 "$1" "$2"
    fi
}

# Used by the back-tracking tracker (core/tracking.py)
clone https://github.com/facebookresearch/sam2.git              sam2

# Used to generate the inputs / other pipeline stages (see README)
clone https://github.com/facebookresearch/co-tracker.git        co-tracker   # point back-tracking
clone https://github.com/ViTAE-Transformer/DeepSolo.git         DeepSolo     # large-text teacher
clone https://github.com/IDEA-Research/GroundingDINO.git        GroundingDINO        # detection / student

echo "Done. See README.md for the expected layout and checkpoint downloads."
