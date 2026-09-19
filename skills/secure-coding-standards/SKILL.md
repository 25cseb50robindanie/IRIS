---
name: secure-coding-standards
description: Security requirements for this codebase — desktop satellite-imagery analysis tool submitted to a defense ministry evaluation. Use whenever writing or reviewing code that handles file I/O, the local backend server, user-provided paths or search input, or Electron window/process configuration. Trigger on any new endpoint, file read/write, or window creation.
---

# Secure coding standards

This codebase is submitted as source code for a Ministry of Defence evaluation. Code must follow established secure-engineering practice, not just "work" — no shortcuts that would embarrass the team if read line by line by a reviewer.

## Input validation

- Validate all data from outside this codebase's own logic before using it: file paths from the folder browser, search query text, imported metadata values.
- Use an allow-list of expected formats/patterns, not a deny-list of known-bad ones.
- All validation failures should result in rejection with a clear error, not a silent fallback.

## File handling

- Never construct a file path by directly concatenating user-provided input. Sanitize and validate paths before any read/write, to prevent path traversal.
- The application only ever reads from user-selected folders/files — never writes outside its own designated app-data directory without an explicit user action.

## Error handling & logging

- Fail safely: catch and handle errors without leaking internal file paths, stack traces, or system details into anything the analyst sees on screen.
- Log enough detail for debugging, but do not log sensitive content (e.g. full imagery data) to plain-text logs unnecessarily.

## Local server (FastAPI / titiler)

- Bind to `localhost` / `127.0.0.1` only. Never `0.0.0.0`. This server must never be reachable from outside the machine it runs on.
- No hardcoded credentials or secrets anywhere in the codebase (not expected to be needed, given the offline design — flag if one appears to be required, that likely signals a design problem).

## Electron-specific

- `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true` on every `BrowserWindow` — these are Electron's own defaults; do not disable them for development convenience and leave them disabled.
- Only expose the specific functions the frontend needs via a `preload` script using `contextBridge` — never expose raw filesystem or process access to the renderer.
- Never load remote or untrusted URLs inside any app window.
- Define a Content-Security-Policy.
- Never set `allowRunningInsecureContent: true` or disable `webSecurity`.
- Do not enable experimental Electron features.

## Dependencies

- Pin dependency versions (lockfiles for both npm and pip) — this also satisfies the submission's reproducibility/provenance requirement.
- Keep Electron itself on a current version — vulnerabilities in the underlying Chromium/Node.js are patched upstream regularly.
- Don't add a dependency without checking it's still maintained.

## General

- Remove dead code, commented-out debug blocks, and any test/mock endpoints before code is considered submission-ready.
- No `eval()`, no dynamic construction of shell commands from user input, no dynamic construction of file-system commands from unsanitized input.
