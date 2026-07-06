# Publication Analysis Scripts

This folder contains analysis scripts organised by research questions.

## Inputs

The scripts expect de-identified CSV inputs:

- encounter-level form data, one row per submitted post-encounter form
- clinician-level eligible ADS note counts
- baseline and follow-up questionnaire exports
- questionnaire composite-score data
- linked note-pair metric data for text carryover analyses

Input paths are supplied by command line. If your local data use different column names, use the column-name options shown by `--help`.

The default public column names are `clinician_id` and `specialty` for clinician-level identifiers, `participant_id` for questionnaire linkage, and neutral ADS/EHR note-pair fields such as `ads_note_id`, `ehr_note_id`, and `initial_draft_to_ehr_reuse_ratio`.

## Scripts

### `00_baseline_characteristics.py`

| Research question | Data source | Method | Output |
|---|---|---|---|
| What were the baseline characteristics of participating clinicians (manuscript Table 1)? | Baseline questionnaire CSV. | Recodes profession, department, sex, age group, clinical language, and free-text dialect into the manuscript Table 1 categories and reports counts and percentages, plus median (IQR) years of clinical experience. | `baseline_characteristics_summary.csv` |

### `01_sample_primary_endpoint.py`

| Research question | Data source | Method | Output |
|---|---|---|---|
| What was the planned minimum sample size for the primary endpoint? | Command-line planning parameters `--planning-rate` and `--planning-half-width`. | Normal-approximation precision formula `ceil(z^2 * p * (1-p) / d^2)` using `z=1.959963984540054`, assumed rate `p`, and half-width `d`. | `sample_primary_endpoint_summary.csv` |
| How many clinicians, submitted encounter forms, and eligible ADS notes were included? | Encounter-level form CSV and clinician-level eligible-note count CSV. | Counts distinct `clinician_id` values, submitted encounter-form rows, and summed eligible ADS-note counts. | `sample_primary_endpoint_summary.csv` |
| What was form coverage among eligible ADS notes? | Encounter-level form CSV and clinician-level eligible-note count CSV. | Divides submitted encounter-form rows by summed eligible ADS-note counts and reports count, percent, and Wilson 95% confidence interval. | `sample_primary_endpoint_summary.csv` |
| What proportion of submitted forms contained at least one major error? | Encounter-level form CSV with a binary major-error field. | Counts forms with a positive major-error flag, divides by all submitted forms, and reports percent plus Wilson 95% confidence interval. | `sample_primary_endpoint_summary.csv` |
| What were the major-error rates by specialty? | Encounter-level form CSV with specialty and binary major-error fields. | Groups forms by specialty, counts positive major-error flags, divides by specialty-specific submitted forms, and reports Wilson 95% confidence intervals. | `primary_endpoint_by_specialty.csv` |
| What proportion of forms contained minor errors or no reported errors? | Encounter-level form CSV with binary major- and minor-error fields. | Counts minor-error forms, major-only forms, minor-only forms, forms with both error types, and forms with neither error type; reports percentages and Wilson 95% confidence intervals. | `sample_primary_endpoint_summary.csv` |
| How was overall draft quality rated? | Encounter-level form CSV with draft-quality ratings. | Recodes quality ratings to ordered English labels and reports counts and percentages, including combined good/very good and poor/very poor rows. | `draft_quality_summary.csv` |
| What clinician-level counts are needed for the primary endpoint figure? | Encounter-level form CSV with clinician, specialty, major-error, and minor-error fields. | Groups forms by clinician and reports submitted forms, major-error forms, minor-error forms, and clinician-level error rates. | `figure_clinician_error_counts.csv` |

### `02_major_error_characterization.py`

| Research question | Data source | Method | Output |
|---|---|---|---|
| Among forms with at least one major error, how severe were the errors after consensus review? | Encounter-level form CSV with binary major-error and consensus severity-score fields. | Restricts to major-error forms, validates severity scores `0` through `5`, and reports count, percent, and count/denominator display by severity score. | `major_error_severity_summary.csv` |
| Which error categories were assigned to major-error forms? | Encounter-level form CSV with binary major-error and semicolon-delimited taxonomy fields. | Restricts to major-error forms, canonicalizes taxonomy tokens, treats categories as non-mutually exclusive, and counts forms in each taxonomy category. | `major_error_taxonomy_summary.csv` |
| Did major errors originate in the transcript, the draft note, both, or remain uncertain? | Encounter-level form CSV with binary major-error and adjudicated origin fields. | Restricts to major-error forms, canonicalizes origin labels, and reports counts and percentages for transcript, draft note, both, uncertain, and missing. | `major_error_origin_summary.csv` |
| How many major errors were reported per major-error form? | Encounter-level form CSV with binary major-error and reported major-error count fields. | Restricts to major-error forms, sums numeric reported major-error counts, counts missing numeric values as one error for the minimum total, and reports the median and first/third quartiles among nonmissing numeric counts. | `major_error_count_summary.csv` |

### `03_questionnaire_outcomes.py`

| Research question | Data source | Method | Output |
|---|---|---|---|
| Did self-reported documentation time change between baseline and follow-up? | Baseline and follow-up questionnaire CSVs linked by `participant_id`. | Uses complete paired responses, extracts numeric values, computes baseline mean, follow-up mean, mean paired change, t-based 95% confidence interval for the mean change, two-sided Wilcoxon signed-rank test, and Benjamini-Hochberg adjusted P value across paired endpoints. | `questionnaire_paired_outcomes.csv` |
| Did NASA-TLX workload items change between baseline and follow-up? | Baseline and follow-up questionnaire CSVs linked by `participant_id`. | Applies the same complete-pair procedure to mental demand, time pressure, effort, frustration, and perceived task performance. | `questionnaire_paired_outcomes.csv` |
| Did consultations per workday change between baseline and follow-up? | Baseline and follow-up questionnaire CSVs linked by `participant_id`. | Applies the same complete-pair procedure to consultations per workday. | `questionnaire_paired_outcomes.csv` |
| What were the post-implementation acceptance and UTAUT domain scores? | Questionnaire composite-score CSV with post-implementation UTAUT items and overall acceptance index. | Restricts to rows marked as post-questionnaire present when that field exists, uses the supplied overall acceptance index, computes respondent-level domain means from item groups, and reports mean and standard deviation. | `questionnaire_post_acceptance.csv` |
| How many respondents agreed that the ADS was useful in practice and easy to use? | Questionnaire composite-score CSV with post-implementation UTAUT item fields. | Counts nonmissing responses with item score `>=4` for the usefulness and ease-of-use items and reports numerator, denominator, and percent. | `questionnaire_post_acceptance.csv` |

### `04_primary_endpoint_sensitivity.py`

| Research question | Data source | Method | Output |
|---|---|---|---|
| How does clinician clustering affect uncertainty around the primary endpoint? | Encounter-level form CSV with clinician identifier and binary major-error field. | Compares the encounter-level major-error estimate using naive binomial variance with an intercept-only ordinary least-squares linear probability model using clinician-cluster-robust standard errors. | `primary_endpoint_cluster_sensitivity.csv` |
| What is the clinician-resampled bootstrap interval for the primary endpoint? | Encounter-level form CSV with clinician identifier and binary major-error field. | Resamples clinicians with replacement, carries all sampled clinician encounters into each replicate, and reports the 2.5th and 97.5th percentiles of the bootstrap encounter-level major-error rate. | `primary_endpoint_cluster_sensitivity.csv` |
| What is the equal-weight-per-clinician sensitivity estimate? | Encounter-level form CSV with clinician identifier and binary major-error field. | Computes each clinician's major-error rate, averages clinician rates with equal clinician weight, and bootstraps this equal-weight estimand by resampling clinicians. | `primary_endpoint_cluster_sensitivity.csv` |
| What clinician-level primary-endpoint counts are used in clustering and bootstrap analyses? | Encounter-level form CSV with clinician, specialty, and binary major-error fields. | Groups forms by clinician and reports forms, major-error forms, and clinician-level major-error rate. | `primary_endpoint_clinician_summary.csv` |
| What were weekly observed error and quality summaries? | Encounter-level form CSV with timestamp, binary major-error, binary minor-error, and quality fields. | Parses form timestamps, defines study week as `floor(days since first form / 7) + 1`, and reports weekly counts/rates for major, minor, and any error plus mean quality score. | `primary_endpoint_weekly_summary.csv` |
| Did the observed major-error rate change over calendar time? | Encounter-level form CSV with clinician identifier, timestamp, and binary major-error field. | Fits a linear probability model with major-error status as the binary outcome and days since first form as a continuous predictor; reports slope, confidence interval, and P value using clinician-cluster-robust standard errors. | `primary_endpoint_time_trend_tests.csv` |
| Did minor-error rate, any-error rate, or quality score change over calendar time? | Encounter-level form CSV with clinician identifier, timestamp, binary error fields, and quality field. | Fits the same clinician-cluster-robust linear trend model separately for minor-error status, any-error status, and numeric quality score. | `primary_endpoint_time_trend_tests.csv` |

### `05_clinician_exploratory_associations.py`

| Research question | Data source | Method | Output |
|---|---|---|---|
| How much did reported major- and minor-error rates vary between clinicians? | Encounter-level form CSV with clinician identifier and binary error fields. | Groups forms by clinician, computes clinician-level major- and minor-error rates with Wilson confidence intervals, and tests common-rate variation using chi-square calculations plus Monte Carlo binomial simulation under a common pooled rate. | `clinician_error_variation_summary.csv`, `clinician_error_variation_tests.csv` |
| Which clinician baseline characteristics were associated with submitted-form major-error or any-error status? | Encounter-level form CSV and baseline questionnaire CSV linked by clinician/participant identifier. | Builds one row per submitted form, links clinician baseline characteristics, and fits separate univariable logistic regression models for major-error status and any-error status using clinician-cluster-robust standard errors. Baseline clinician characteristics are not combined in one adjusted model because the exploratory data set has few major-error events and few clinician clusters. | `baseline_logistic_regression_models.csv`, `baseline_logistic_regression_table.csv`, `baseline_logistic_regression_table_preview.md` |
| Was form-submission adherence associated with observed error rates? | Encounter-level form CSV and clinician-level eligible-note count CSV. | Computes clinician-level form-submission adherence as submitted forms divided by eligible ADS notes, then applies Spearman correlation between adherence and clinician-level major-, minor-, and any-error rates. | `form_adherence_user_level.csv`, `form_adherence_bias_tests.csv` |
| Did lower-adherence clinician groups have different observed error proportions? | Encounter-level form CSV and clinician-level eligible-note count CSV. | Splits clinicians by adherence `<0.5` versus `>=0.5` and by `<=median` versus `>median`, pools forms within each split, and applies two-sided Fisher exact tests for major-, minor-, and any-error forms. | `form_adherence_bias_summary.csv`, `form_adherence_bias_tests.csv` |

### `06_text_carryover.py`

| Research question | Data source | Method | Output |
|---|---|---|---|
| How many submitted forms had linkable ADS and EHR note-pair metrics? | Linked note-pair metric CSV with clinician identifier and note-pair identifiers. | Counts linked metric rows and distinct clinicians. | `text_carryover_overall_summary.csv` |
| What proportion of final EHR-note text could be traced to the initial ADS draft? | Linked note-pair metric CSV with `initial_draft_to_ehr_reuse_ratio` or the configured initial-draft metric. | Summarizes nonmissing metric values with mean, median, first quartile, third quartile, interquartile range, minimum, maximum, share `>=0.60`, and count/share strictly `>0.90`. The denominator is the final EHR note, so this does not estimate the proportion of draft text that remained unchanged. | `text_carryover_overall_summary.csv` |
| How many rows and unique note pairings met the configured initial-draft carryover threshold? | Linked note-pair metric CSV with note-pair identifiers and the initial-draft-to-EHR metric. | Counts rows and unique ADS/EHR note pairings with metric values greater than or equal to the configured threshold, and counts clinicians with at least one row meeting the threshold. The default threshold is `0.60`. | `text_carryover_threshold_summary.csv` |
| How did threshold counts vary by specialty? | Linked note-pair metric CSV with specialty and initial-draft-to-EHR metric fields. | Repeats the threshold row, unique-pairing, and clinician counts within each specialty. | `text_carryover_threshold_summary.csv` |

## Running the scripts

Install dependencies:

```bash
python3.13 -m pip install -r requirements.txt
```

Template:

```bash
python3.13 01_sample_primary_endpoint.py \
  --encounters path/to/main_encounters.csv \
  --eligible-notes path/to/eligible_note_counts.csv \
  --output-dir outputs/primary_endpoint
```

## Outputs

Each script writes one or more aggregate CSV files to the requested output directory.

## Reproducibility

The scripts were written for Python 3.13 and use the packages listed in `requirements.txt`.
