# TLR8 potentiation case: a sequence-context story within one fixed chemistry

## Reader-level conclusion

For `GTCCGTACCTCCACCCACCG`, the model does not identify a single hazardous motif. Within the Alharbi 2020 fixed 5–10–5 2′-O-methyl/DNA-gap/full-PS assay context, the C-rich, low-T sequence with no TT is placed in the low-to-moderate TLR8-potentiation range. The measured reporter response is 1.70; 15 outer-held-out members that saw neither the row, HELM nor canonical sequence predicted 1.271.

## Data and scope

The labels are the 192-ASO Alharbi et al. screen at 100 nM in an HEK293-TLR8 NF-κB reporter with R848 normalized to 1.0. Every molecule has modification pattern `OOOOODDDDDDDDDDOOOOO`; the endpoint can therefore resolve sequence context within this architecture but cannot identify the effect of changing to LNA or MOE.

The 1,918 descriptors are not copied from one paper. Raw positional tokens follow HELM and OligoGym representation; composition/k-mer baselines follow the OligoGym design; wing/gap and sequence-by-chemistry interactions are project-engineered columns motivated by the Papargyri, Bhamra and Alharbi experimental designs. Papers motivate why a molecular fact is represented, whereas project code defines its exact numerical derivation. The Supplement feature-level provenance database records that distinction, its implementation path and source locator.

## Endpoint trend and query interpretation

Across the complete source endpoint, descriptive Spearman associations are 0.468 for T fraction, 0.420 for TT count, -0.466 for G fraction and -0.429 for GG count. The query has T fraction 0.15, TT=0 and C fraction 0.55. It contains 2 TCC motifs, but TCC count alone has Spearman -0.104 (unadjusted P=0.152). Thus the readable signal is joint composition and position context, not a TCC-count rule.

## Same-model integration and deletion evidence

Relative to a mean member baseline of 2.221, motif/context, Registry tokens and sequence-by-position/chemistry descriptors contribute -0.514, -0.300 and -0.136, respectively, reconciling to 1.271. Removing and refitting the motif block raises the case prediction to 1.695, showing that sequence-context dependence explains most of this underprediction. Endpoint Spearman falls from 0.709 to 0.643 without motif and 0.640 without Registry, whereas removing the chemistry-interaction block gives 0.726. The latter is a warning against claiming an independently learned chemistry mechanism in a fixed-chemistry dataset.

## Analogue contrast and falsification

The nearest measured analogue has the same chemistry, 15% T, no TT, two TCC motifs and response 1.00. The informative contrast also has two TCC motifs but 30% T, one TT and response 3.95. This supports a testable sequence-context contrast, not shared mechanism. Composition-matched reciprocal relocation of TT/T-rich context between wing and gap, plus TCC-disrupting controls, should be tested with TLR8 and receptor-negative reporters, uptake and viability. Reporter association, TreeSHAP, deletion and similarity remain non-causal evidence types.
