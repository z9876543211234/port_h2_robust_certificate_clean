# Frozen input hashes

## Common 20 MW / 34-AGV inputs

| File | SHA-256 |
|---|---|
| `C1_base20mw_agv34.yaml` | `41b2b660f9d8e6946bb9b31913dc9568f8844b4afeb881638c41e5fca2278dfe` |
| `physical_large_port_96.json` | `9587537cac53e9ede53c0c837b0f1e7f794b6e039361703abe8c3761d415c863` |
| `profile_96.yaml` | `a6c3d27596f818f0687fb00512640093c6a76bb3243cf9c1908d6ed6350a8def` |
| `wind_large_port_96.json` | `7224f6e0b8334af44ae971617bcbd687f1abb730418796b56eaa7f2dc1ba9a71` |
| `ship_delay_large_port_96.json` | `1bd811a702df455041ac80a6cb4185f7f6842160112e7e348ccb17d82e2fd0f8` |

The profile, wind, and ship-delay hashes are byte-for-byte identical to the
preserved 40 MW / 57-AGV input package.  The deterministic file changes only
the 96-period base-load series from 40,000 to 20,000 kW, the AGV fleet from 57
to 34, and the associated provenance fields.  All other physical and economic
parameters are retained.

## Loaded C1 identity

- Case semantic SHA-256:
  `945e9596b420d6832aed410a63f2dffe0686097251569093e4745d0e7cdf28d2`
- Joint uncertainty Bundle SHA-256:
  `f1af6d2a2d5c6d4ab362891dfd84a122e0f0ff802e2946b11a30ae4633749e53`
- Formal input validation: passed.

## Selected C3 identity

The complete 50%--85% scan selected an 85% container share.  With 34 AGVs,
half-up rounding gives an integer 29-vehicle container group and a 5-vehicle
LOHC group.

- Selected case file: `C3_base20mw_agv34_selected.yaml`
- Selected case-file SHA-256:
  `11bc1b504cbf849fab67a791310c05436c16187dae58e1fa532f57765db585fb`
- Case semantic SHA-256:
  `cd21074f5fa52f26a9af4b2cfd889441a83b33d94af038639bccdf85133306ca`
- Joint uncertainty Bundle SHA-256:
  `f1af6d2a2d5c6d4ab362891dfd84a122e0f0ff802e2946b11a30ae4633749e53`
- Formal input validation: passed.

The selected file is byte-for-byte identical to
`C3_base20mw_agv34_ctn29_lohc5.yaml`.  The seven other candidate case files
and their scan results remain in the same isolated experiment for audit.

## C4 and C2 identities

Both cases reuse the exact same 20 MW / 34-AGV deterministic file and the
same profile, wind, and ship-delay exogenous inputs listed above.

| Case | Case file SHA-256 | Case semantic SHA-256 | Validation |
|---|---|---|---|
| C4 deterministic | `c9a96e51355fb6a8f676d545889d5666becc9477514e586068de6a9d903c440c` | `bdbeb81774ff8d1602d14708b9c7a8956899bb4753df36357a5ac80a74992db2` | passed |
| C2 no-hydrogen robust | `0445f9ebb52db5931ffc5d7acdbc50672b5ed316089a7f4c41bfcccbd53a8220` | `cd985ab2714ff2df73b36d99ecec58b95be7a6ed97a11d0da21c4450f8094ccc` | passed |

The joint uncertainty Bundle SHA-256 remains
`f1af6d2a2d5c6d4ab362891dfd84a122e0f0ff802e2946b11a30ae4633749e53`
for both cases.  C2 structurally disables the hydrogen and LOHC chain; C4
retains it but optimizes the nominal deterministic case before the formal
stress replay defined by the existing C4 solver path.
