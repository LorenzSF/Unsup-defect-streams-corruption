---
name: OnDemand web terminal mangles long pasted commands
description: How to write multi-line files reliably inside the KU Leuven OnDemand web shell
type: feedback
---

The OnDemand web terminal at https://ondemand.hpc.kuleuven.be silently mangles pasted multi-line commands and heredocs. Symptoms observed in 2026-04-25 session:

- A `cat > file <<'EOF' ... EOF` heredoc was written **verbatim** into the file (the `cat`, `<<'EOF'`, and `EOF` lines all ended up as file contents). Bash never executed it.
- A long single-line `printf '...' > file` got mangled the same way: the printf command itself ended up as the file's content.
- Backslash line-continuations (`\` at end of line) silently dropped, so each `--flag` line became a separate command and `bash: --config: command not found` errors fired.
- Ctrl+C inside a stuck heredoc terminated the SSH connection entirely, requiring a session reconnect.

**Why:** unclear, likely a paste-buffering or bracketed-paste bug in the OnDemand terminal frontend. Reproduced multiple times in one session. Independent of remote shell state.

**How to apply:** when working through this terminal:

1. **Never paste heredocs.** Use `nano` instead — open the file, paste content into the editor (interactive, no shell interpretation), Ctrl+O Enter Ctrl+X.
2. **Avoid long quoted single-line commands.** Long `printf '...' > file` invocations get mangled too.
3. **Prefer short `sed -i 's|old|new|' file` edits** for changing existing config files. One short command per change. This worked reliably when heredocs and printf failed.
4. **Avoid backslash continuations** in pasted commands — put the whole command on one short line if it fits, otherwise use a saved script file.
5. **Use short relative paths** by `cd`-ing into the working dir first. Long absolute paths in commands are more likely to be split/duplicated by paste.

The local terminal launched via `Open Terminal` in OnDemand sessions is the affected one. Heredocs work fine in real SSH shells when the user has VPN access — but the user does not have KU Leuven VPN, so OnDemand is the only option.
