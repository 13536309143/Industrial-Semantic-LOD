# README figure provenance

The figures in this directory support the English and Chinese READMEs.

| Asset | Source | Meaning |
| :--- | :--- | :--- |
| `industrial-assembly.png` | Supplied `10.8图片.pptx`, slide 2, embedded `ppt/media/image1.png` | Industrial assembly illustration; not a renderer capture |
| `structure-preservation.png` | Supplied `10.8图片.pptx`, slide 6, embedded `ppt/media/image139.png` | Conceptual structure-preservation illustration; not measured simplification output |
| `pointnext-training.png` | Recorded industrial-part `log.txt` | Training OA and evaluation OA at actual five-epoch evaluation intervals |
| `policy-comparison.png` | `analysis_reports/lod_ablation_aggregate_summary.csv` | Mesh-record shares in three protection ranges across six configurations |

The two embedded illustrations were copied without modification. Their original authorship or external provenance is not established by their presence in the supplied presentation. They are credited as supplied figure materials, not as original rendering output of this repository.

The charts were generated from existing results during the documentation review on 2026-10-14. No synthetic scores, new training results, or benchmark speedups are included. The training run uses the same split for validation and test. The policy chart describes allocations, not geometric accuracy or actual culling.

[Evidence metadata](../data/readme-evidence.json) includes source hashes, source geometry counts, classification results, and row/unique-mesh coverage checks for all 30 policy CSVs. Large input scenes and raw policy outputs remain in the ignored local directories.
