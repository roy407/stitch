#!/bin/sh
# scripts/install-git-hooks.sh

ln -sf ../../scripts/git-hooks/commit-msg .git/hooks/commit-msg
chmod +x scripts/git-hooks/commit-msg