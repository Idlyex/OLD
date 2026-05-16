---
name: master-git-automation
description: Provide elite-level, professional, and extremely safe Git and GitHub operations management. Automatically activate only upon detection of specific trigger keywords to perform high-quality commits, pushes, rollbacks, branching strategies, comparisons, pull requests, and other Git operations. Always follows Conventional Commits standards, writes exceptional semantic commit messages, ensures maximum safety, maintains clean and professional git history, and operates with production-grade discipline.
---

Master Git Automation
Overview
This skill provides precise, professional, and highly disciplined control over all Git and GitHub operations. It remains completely passive by default and activates only when specific trigger keywords are detected in the user's message. When activated, it performs all operations at a Staff-level engineering standard — thoughtful, clean, safe, and focused on long-term repository health.

When to Use This Skill
This skill is strictly passive and activates exclusively when the user includes one or more of the following trigger keywords or phrases:

Trigger Keywords:

commit, save changes, stash, save work
push, upload, sync, update remote, publish
revert, rollback, undo, restore, reset, go back
diff, compare, difference, what changed
branch, new branch, switch branch, create branch
merge, pull request, pr, create pr, review pr
pull, fetch, update, rebase
status, log, history, show changes
Core Capabilities

1. Intelligent Commit Management
Automatically analyze changes, group them logically, stage appropriate files, and create high-quality Conventional Commits with excellent semantic messages.

Apply when:

User wants to commit changes
Changes need to be organized and documented properly
Commit framework:

Analyze what files were changed and why
Group related changes into logical commits
Never commit unrelated or temporary files
Write clear, informative, and professional commit messages following Conventional Commits specification (feat:, fix:, refactor:, perf:, docs:, chore:, test:, build:, etc.)
2. Smart Push Logic
Decide intelligently whether to push after commit, create new branches when appropriate, and handle remote synchronization safely.

Apply when:

User wants to push changes
Work needs to be shared or backed up
Push framework:

Check if commit is needed first
Determine correct branch strategy
Push to appropriate remote
Create upstream branch when necessary
Handle force push only with explicit confirmation
3. Safe Rollback & Recovery Operations
Perform safe and reversible recovery operations with full visibility of consequences.

Apply when:

User wants to undo, revert, or restore previous state
Rollback framework:

Show exact diff of what will change
Offer safest possible method first (revert over reset)
Require explicit confirmation for destructive actions
Provide recovery options if something goes wrong
4. Comparison & Diff Analysis
Provide clear, meaningful, and actionable comparisons between commits, branches, or files.

Apply when:

User wants to compare changes or understand differences
Comparison framework:

Summarize important changes at high level
Highlight breaking changes and critical modifications
Explain impact on architecture or performance
Provide context for why changes were made
5. Branch & Workflow Management
Create properly named branches and manage Git workflows according to best practices.

Apply when:

User wants to create branches or manage workflow
Branching framework:

Use semantic branch naming (feature/, fix/, refactor/, perf/, docs/, experiment/)
Follow Git Flow or Trunk-Based Development as appropriate
Handle merging strategies safely
Support pull request creation with proper description
6. High-Quality Commit Messages & Documentation
Always produce professional, informative, and useful commit messages and PR descriptions.

Commit Message Standards:

Use Conventional Commits format
Include clear subject line (max 50 characters)
Provide detailed body when necessary
Reference issues or tickets when relevant
Explain "what" and "why", not just "how"
Application Guidelines

Response Structure (when skill is triggered):

Understanding
Clear summary of what the user wants to accomplish with Git.

Proposed Action Plan
Exact sequence of Git commands that will be executed.

Commit Message Proposal (if commit is involved)
Full, ready-to-use commit message following standards.

Impact Analysis
What files will be affected, potential risks, and consequences.

Safety Confirmation
For any potentially dangerous operation (reset, force push, clean, etc.) — always ask for explicit confirmation with clear warning.

Execution
After confirmation — execute cleanly and report results.

Safety Rules (Never Break These):

Never perform force push without explicit user permission
Never execute destructive commands (hard reset, clean -f, etc.) without clear confirmation
Never commit generated files, logs, node_modules, or large binaries unless explicitly instructed
Always show the user what will happen before doing anything irreversible
Always prioritize repository history cleanliness and team safety
Examples of Smart Behavior:

User says "commit my changes" → Analyzes changes, creates logical commits with excellent messages
User says "push this" → Commits if needed, then pushes safely
User says "rollback last commit" → Suggests safe revert first, shows diff
User says "compare with main" → Provides meaningful summary of differences
User says "create new feature branch" → Creates properly named branch and switches to it
Remember

Git history is a permanent record. Treat it with respect.
Quality and safety are more important than speed.
A clean git history is a sign of professional engineering.
Every commit should make the repository better, never worse.
The goal is to make Git operations feel effortless, safe, and professional for the user.
Quick Git Operation Checklist

Before Any Action (5 points)

 Are trigger keywords properly detected?
 Is the requested operation safe?
 Have I shown the user what will happen?
 Is the commit message high quality?
 Have I considered long-term repository health?
Score interpretation:
5 = Perfect execution
4 = Good with minor issues
≤3 = Requires improvement — safety or quality compromised