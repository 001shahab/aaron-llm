#!/usr/bin/env python3
# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Assert that the whole history is single authored.

The commit-msg hook in ``.githooks`` enforces this on the way in, but a hook lives in
a clone and can be missing, bypassed with ``--no-verify``, or skipped by a rebase that
replays a commit unchanged. This script checks the objects that actually exist, so the
guarantee does not depend on anyone's local setup. CI runs it on every push.

Checked, across every commit reachable from any ref, plus every tag:

- author and committer are Prof. Shahab Anbarjafari <shb@3sholding.com>
- author and committer are identical
- no attribution trailer in any commit or tag message
- no tool credit written into a message as prose

Usage:

    python3 scripts/check_authorship.py
"""

from __future__ import annotations

import re
import subprocess
import sys

EXPECTED = "Prof. Shahab Anbarjafari <shb@3sholding.com>"

TRAILER = re.compile(
    r"^[ \t]*(co-authored-by|signed-off-by|assisted-by|reviewed-by|helped-by"
    r"|generated-by|created-by)[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)

# Attribution phrasing only. A model named as the subject of a change, for example
# "fix the gpt-4o cost calculation", is ordinary content in this repository.
CREDIT = re.compile(
    r"generated (with|by)"
    r"|co-?(written|created|developed) (with|by)"
    r"|(written|created|authored|produced|implemented|assisted) (with|by) +(an? +)?"
    r"(ai|llm|assistant|agent|claude|copilot|chatgpt|cursor|gemini|codex)",
    re.IGNORECASE,
)

SEPARATOR = "\x1e"


def git(*args: str) -> str:
    """Run a git command and return its output, or exit if git itself fails."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, encoding="utf-8"
    )
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def check_commits(problems: list[str]) -> int:
    """Check the identity and message of every commit reachable from any ref."""
    fields = f"%H{SEPARATOR}%an <%ae>{SEPARATOR}%cn <%ce>{SEPARATOR}%B"
    log = git("log", "--all", "--no-merges", f"--format={fields}%x00")
    commits = [entry for entry in log.split("\0") if entry.strip()]

    for entry in commits:
        sha, author, committer, message = entry.strip("\n").split(SEPARATOR, 3)
        short = sha[:9]
        if author != EXPECTED:
            problems.append(f"{short} author is {author}, expected {EXPECTED}")
        if committer != EXPECTED:
            problems.append(f"{short} committer is {committer}, expected {EXPECTED}")
        found = TRAILER.search(message)
        if found:
            line = message[found.start() : message.find("\n", found.start())].strip()
            problems.append(f"{short} message carries an attribution trailer: {line!r}")
        credit = CREDIT.search(message)
        if credit:
            problems.append(f"{short} message credits a tool: {credit.group(0)!r}")
    return len(commits)


def check_tags(problems: list[str]) -> int:
    """Check the tagger and message of every annotated tag."""
    fields = f"%(refname:short){SEPARATOR}%(taggername) %(taggeremail){SEPARATOR}%(contents)"
    listing = git("for-each-ref", "refs/tags", f"--format={fields}%00")
    tags = [entry for entry in listing.split("\0") if entry.strip()]

    for entry in tags:
        name, tagger, message = entry.strip("\n").split(SEPARATOR, 2)
        if tagger.strip() and tagger.strip() != EXPECTED:
            problems.append(f"tag {name} tagger is {tagger.strip()}, expected {EXPECTED}")
        if TRAILER.search(message):
            problems.append(f"tag {name} message carries an attribution trailer")
        if CREDIT.search(message):
            problems.append(f"tag {name} message credits a tool")
    return len(tags)


def main() -> int:
    """Report every authorship problem found, and exit non zero if there is one."""
    problems: list[str] = []
    commits = check_commits(problems)
    tags = check_tags(problems)

    if problems:
        print(f"authorship: {len(problems)} problem(s) found")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nFix with an amend or an interactive rebase so the history holds exactly "
            "one author and one committer. See .cursor/rules/solo-authorship.mdc."
        )
        return 1

    print(f"authorship: {commits} commits and {tags} tag(s), all by {EXPECTED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
