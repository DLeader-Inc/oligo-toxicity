# Licence boundaries and source attribution

`LICENSE` applies to new project-owned Python code. It does not relicense
third-party source code, datasets, saved measurement labels or model weights.
Dependencies retain their own licences. OligoGym is fetched separately from
Roche/OligoGym commit `97f5b9f58d9e445a8ecb0218037af7465c3708c0` under
its Apache-2.0 repository licence. Preserve that repository's notices.

Source data remain at their original hosts. Downloading is opt-in through
`reproduce/download_data.py`, which verifies fixed revisions and SHA-256.

| Endpoint | Data source | Release weight terms |
|---|---|---|
| HeLa caspase-3/7 | Papargyri et al., 2020, DOI 10.1016/j.omtn.2019.12.011, CC BY 4.0 article | CC BY 4.0 |
| LNA neuronal calcium | Hagedorn et al., 2022, DOI 10.1089/nat.2021.0071, CC BY 4.0 article | CC BY 4.0 |
| MOE mouse ICV FOB | 13-patent collection packaged by OligoGym, Apache-2.0 repository | Apache-2.0, as selected by the authors |
| TLR8 potentiation and TLR7 inhibition | Alharbi et al., 2020, DOI 10.1093/nar/gkaa523, CC BY-NC 4.0 | CC BY-NC 4.0 |
| Mouse/rat ALT and FOB | ASO Atlas 2 processed dataset, CC BY 4.0, revision `b947db330a3bf1e194ac190d0f188f7693660852` | CC BY 4.0 |

The legacy code keys `tlr7_immunotox` and `tlr8_immunotox` refer to TLR8
potentiation and TLR7 inhibition, respectively. Canonical biological names
are used in the public prediction output. This reversal is documented rather
than silently changing the mapping of trained weights.

Archived tables and predictions under `results/` and `outputs/` include
third-party-derived evidence and are not MIT software. Use them for
noncommercial research replication with attribution and retain the upstream
conditions, including CC BY-NC 4.0 for TLR-derived records. This mixed archive
does not grant unrestricted commercial use of TLR data or derived models.
The authors' own contributions to that mixed evidence archive are offered
under CC BY-NC 4.0, subject to existing upstream rights.

Source provenance and row counts are preserved in
`results/supporting/training_data_provenance.csv`. Source papers, dataset
licences, and patent rights address different rights and must not be
conflated. Public patent disclosure does not itself grant freedom to practise
a patented therapeutic invention. Model licence choices are release terms,
not a legal conclusion that training automatically transfers data licences
to every trained weight.

Licence texts: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en),
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
