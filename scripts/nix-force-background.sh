#!/bin/bash
# Force nix commands to run in background to avoid token waste during long builds.
#
# This hook intercepts Bash(nix *) calls and sets run_in_background: true,
# then informs Claude to use Monitor + ScheduleWakeup to avoid blocking.
#
# Usage: Invoked automatically by PreToolUse hook in .claude/settings.json
# Hook input: JSON from Claude Code containing tool_input.command and tool_input

set -e

input=$(cat)

# Extract the command from hook input JSON
command=$(echo "$input" | jq -r '.tool_input.command // empty')

if [[ -z "$command" ]]; then
  # Not a valid hook input, pass through unchanged
  echo "$input"
  exit 0
fi

# Only process nix commands (the matcher should already filter, but double-check)
if [[ "$command" =~ ^nix ]]; then
  # Modify the tool input to force background execution
  echo "$input" | jq \
    --arg ctx "This nix build is running in background to save tokens. Use Monitor to stream output, then ScheduleWakeup to sleep until complete." \
    '.tool_input.run_in_background = true |
     .hookSpecificOutput = {
       "hookEventName": "PreToolUse",
       "additionalContext": $ctx
     }' \
    2>/dev/null || echo "$input"
else
  # Not a nix command, pass through unchanged
  echo "$input"
fi
