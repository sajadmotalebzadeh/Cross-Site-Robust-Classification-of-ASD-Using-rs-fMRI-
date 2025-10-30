# Cross-Site-Robust-Classification-of-ASD-Using-rs-fMRI-
Cross-Site Robust Classification of Autism Spectrum Disorder Using rs-fMRI with Dual-Attention Deep Learning and Calibrated XGBoost on ABIDE-I
Robust Autism Classification on ABIDE-I

Dual-Attention Neural Network + Calibrated XGBoost (site-grouped CV)

End-to-end, leakage-free pipeline for ASD vs. control on ABIDE-I, with figure generation (Figs 1–8), manual probability calibration, and outer GroupKFold by SITE_ID.

⸻

Overview

This repo implements a fusion approach:
	•	Imaging stream: rs-fMRI ⟶ CC200 ROI time series ⟶ FC (Pearson) ⟶ Fisher’s z ⟶ PCA (95% var, ≤200 PCs)
	•	Phenotype stream: Phenotypic variables ⟶ encode / impute / scale
	•	Stacking: DA-NN (PyTorch) + XGBoost, manual calibration (isotonic or Platt), then a logistic meta-learner with a tuned threshold (default: F1)
	•	Evaluation: outer GroupKFold by SITE_ID; inner split for early stopping, calibration, and threshold selection
	•	No leakage: all preprocessing is fit on training only; SITE_ID is not a feature; One-Hot categories are globally fixed so the design matrix has identical width across folds

⸻

Data
	•	ABIDE-I (public, de-identified): rs-fMRI + phenotypes (obtain under ABIDE terms).
	•	CC200 atlas: Craddock 200-region functional atlas for ROI time series.
	•	This repo does not include raw imaging or raw ROI time series.
	•	We recommend archiving/sharing only derived, non-identifiable artifacts:
	•	subjects.csv (SUB_ID, SITE_ID, label)
	•	outer_folds.csv (SUB_ID → outer fold id)
	•	phenotype_design_matrix.npz (encoded/imputed/scaled phenotypes)
	•	conn_pca_features.npz (FC→z→PCA features; ≤200 PCs)

If you need to regenerate ROI time series: preprocess rs-fMRI to MNI, resample CC200 to functional space, average voxels per ROI → save T×200 matrices. File names must include a parseable subject ID (e.g., SUB_ID_#### or a 5–7 digit ID).

⸻

Repo layout

ASD/
├─ asd_pipline_abide_connectome.py      # main pipeline + figures
├─ env/
│  ├─ environment.yml                   # conda env
│  └─ requirements.txt                  # pip env
├─ (your data) Phenotypic_V1_0b.csv
├─ (your data) abide_cc200_raw/         # CC200 ROI time series (optional)
└─ (outputs) paper_figs_*.png/svg       # figures auto-generated


⸻

Environment

Conda (recommended)

conda create -n asd python=3.12 -y
conda activate asd
pip install numpy pandas matplotlib scikit-learn xgboost torch category-encoders

Pip (alt.)

pip install -r env/requirements.txt


⸻

Quickstart

Phenotype-only (sanity check)

python asd_pipline_abide_connectome.py \
  --csv Phenotypic_V1_0b.csv \
  --out_prefix paper_figs --splits 5 --calibration isotonic --tune_metric f1

Full fusion (phenotypes + CC200 connectome)

python asd_pipline_abide_connectome.py \
  --csv Phenotypic_V1_0b.csv \
  --add_connectivity --ts_dir abide_cc200_raw \
  --out_prefix paper_figs --splits 5 --calibration isotonic --tune_metric f1

Key flags
	•	--splits outer folds (default 5)
	•	--calibration {isotonic|sigmoid|none} (manual; no CalibratedClassifierCV)
	•	--tune_metric {acc|bal_acc|youden|mcc|f1} (default f1)
	•	--pca_var / --pca_max_components for connectome PCA (default 0.95 / 200)

⸻

Outputs (maps to manuscript figures)
	•	figure1_pipeline.svg/png — Overall pipeline (stacking + calibration)
	•	figure2_preprocessing.svg/png — Preprocessing flow (CC200 → FC → z → PCA + phenotype encode → fusion)
	•	figure3_evaluation.svg/png — Evaluation framework (site-grouped CV; inner ES/calibration/threshold)
	•	figure4_roc.png — ROC per fold (thin), mean (bold), ±1 SD
	•	figure5_confusion.png — Confusion matrix (counts + row %)
	•	figure6_xgb_importance.png — XGBoost top-15 (mean gain ± SD across folds)
	•	figure7_nn_loss.png — NN training loss per fold (● best epoch)
	•	figure8_nn_val_auc.png — NN validation AUC per fold (● best epoch)
	•	supp_calibration.png — Reliability diagram (Brier, ECE/MCE + histogram)

Metrics printed: AUC, Accuracy, Balanced Acc, Precision, Recall, F1, MCC.

⸻

Reproducibility & leakage control (what we freeze)
	•	Outer CV: 5-fold GroupKFold by SITE_ID.
	•	Inner split: early stopping (DA-NN), manual calibration (isotonic/Platt) for XGBoost, meta-learner fit, threshold tuning (e.g., F1).
	•	Preprocessing: impute/encode/scale/PCA fit on training only. SITE_ID never used as a feature.
	•	One-Hot stability: global category set fixed → identical feature width across folds.
	•	Seeds: fixed for NumPy, PyTorch, XGBoost (see script).
	•	Artifacts to share: outer fold files, encoded phenotypes, PCA connectome features.

⸻

Troubleshooting
	•	Mac/py3.12/sklearn 1.6 “CalibratedClassifierCV” errors
This repo uses manual calibration (isotonic/Platt) — there should be no CalibratedClassifierCV calls.
If you edited locally, ensure:

grep -n "CalibratedClassifierCV\|xgb_cal_full" asd_pipline_abide_connectome.py

returns nothing. Remove __pycache__/ and rerun.

	•	“Found input variables with inconsistent numbers of samples”
After merging connectomes, only subjects with both phenotypes and ROI time series remain. The script re-aligns labels after merge. Ensure ROI filenames expose SUB_ID (adjust the parser if needed).
	•	Importance vstack dimension mismatch
Fixed by global OHE categories; we also pad as a safety guard.

⸻

CC200 & ABIDE references (cite in your paper)
	•	Craddock RC, James GA, Holtzheimer PE, Hu XP, Mayberg HS. A whole-brain fMRI atlas via spectral clustering. Hum Brain Mapp. 2012;33(8):1914–1928. (CC200)
	•	Di Martino A, Yan CG, Li Q, et al. The autism brain imaging data exchange… Mol Psychiatry. 2014;19:659–667. (ABIDE-I)

⸻

License
	•	Code: MIT (or Apache-2.0) — pick one and include LICENSE.
	•	Data derivatives: share under a permissive data license (e.g., CC BY 4.0) subject to ABIDE terms. Raw imaging and raw ROI time series are not redistributed.

⸻

Citation (template)

If you use this code, please cite your paper and the resources above. For now:

@misc{your_repo_yyyy,
  title  = {Robust Autism Classification on ABIDE-I: Dual-Attention NN + Calibrated XGBoost},
  author = {Your Name},
  year   = {2025},
  url    = {<repo URL>},
  note   = {Versioned snapshot: <DOI or tag>}
}


⸻

Questions or issues? Open an issue with your command line, OS, Python/package versions, and the console log snippet.
