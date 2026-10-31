# Data provenance

GB1 measurements originate in Wu et al., *Adaptation in protein fitness landscapes
is facilitated by indirect paths*, eLife 2016, DOI
[10.7554/eLife.16965](https://doi.org/10.7554/eLife.16965).

This project downloads the full four-site table distributed by
[FLIP](https://github.com/J-SNACKKB/FLIP/tree/main/splits/gb1), pinned to commit
`9f58a97455d4133669c04b0a723f9d6e49e29eb2`. Archive SHA256:
`85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff`.

FLIP attributes the original eLife data to CC BY 4.0 and its modified dataset to
AFL-3.0. The complete source table is downloaded locally and is not redistributed
in this repository. Run artifacts contain queried subsets and derived summaries,
with the same source attribution. The code's MIT license does not supersede
the data terms, which continue to apply to those records.

Two details matter:

- FLIP's `keep` flag is derived from measured fitness to balance the benchmark.
  We ignore it and use the full measured landscape. This is a different protocol
  from published FLIP prediction splits; results are not directly comparable.
- The supplied `sequence` column describes a longer fusion construct. We use
  the explicit four-residue `Variants` key and reconstruct the 56-aa GB1 domain.

Fitness is the published relative IgG-binding enrichment, with the assay wild
type set to 1. It is not a binding constant, a folding measurement, a drug
efficacy endpoint, or a measurement of whole-protein design success. Low-read-count
variants can be noisy; the audit reports their prevalence.

Optional GPU pretraining downloads a current-release UniProt/Swiss-Prot FASTA
from the official UniProt FTP server. Its SHA256 is saved in the run manifest.
The exact corpus bytes are identified by that digest rather than by a claim
that a mutable current-release URL is permanent. UniProt data retain their
[CC BY 4.0 terms](https://www.uniprot.org/help/license). The reservoir sample
uses unambiguous canonical amino acids and 32-4096-residue proteins.

