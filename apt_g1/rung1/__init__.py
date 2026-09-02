"""TO41 Rung 1 Mode A conditioning runtime + independent D checker.

Layout (TO41_D_DRYRUN_PROTOCOL.md §10.2, SCRIPT_MAP registered):

- ``mode_a_runtime``  state-changing execution code (Mode A executor; its CLI
                       covers static coverage / 28-cell decode-only dry-run).
- ``d_checker``       read-only audit (independent; never imports runtime).
- ``rung1_selftest``  local self-test: lookup unit tests + negative tests A-D.

Protocol is FROZEN (refine-logs/TO41_D_DRYRUN_PROTOCOL.md); the only allowed
change after freeze = spec -> code. Any field that would need reinterpretation
=> stop => genuine incompatibility => owner reopen (protocol §7.3).
"""
