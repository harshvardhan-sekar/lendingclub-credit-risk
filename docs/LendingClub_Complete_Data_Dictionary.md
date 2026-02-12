# LendingClub Dataset — Complete Variable Reference

**Source:** Official LCDataDictionary.xlsx + Dataset Profiling (100,000 rows)

**Total Variables:** 151 columns across 8 categories

---

## 1. Loan Information

| Column Name | Definition | Type | % Missing | Notes |
|---|---|---|---|---|
| `desc` | Loan description provided by the borrower | object | 100.0% | Sparse |
| `funded_amnt` | The total amount committed to that loan at that point in time. | float64 | 0.0% | — |
| `funded_amnt_inv` | The total amount committed by investors for that loan at that point in time. | float64 | 0.0% | — |
| `grade` | LC assigned loan grade | object | 0.0% | — |
| `initial_list_status` | The initial listing status of the loan. Possible values are – W, F | object | 0.0% | — |
| `installment` | The monthly payment owed by the borrower if the loan originates. | float64 | 0.0% | — |
| `int_rate` | Interest Rate on the loan | float64 | 0.0% | — |
| `issue_d` | The month which the loan was funded | object | 0.0% | — |
| `loan_amnt` | The listed amount of the loan applied for by the borrower. If at some point in time, the credit d... | float64 | 0.0% | — |
| `purpose` | A category provided by the borrower for the loan request.  | object | 0.0% | — |
| `pymnt_plan` | Indicates if a payment plan has been put in place for the loan | object | 0.0% | — |
| `sub_grade` | LC assigned loan subgrade | object | 0.0% | — |
| `term` | The number of payments on the loan. Values are in months and can be either 36 or 60. | object | 0.0% | — |
| `title` | The loan title provided by the borrower | object | 0.1% | — |
| `url` | URL for the LC page with listing data. | object | 0.0% | — |

## 2. Borrower Demographics

| Column Name | Definition | Type | % Missing | Notes |
|---|---|---|---|---|
| `addr_state` | The state provided by the borrower in the loan application | object | 0.0% | — |
| `annual_inc` | The annual income provided by the borrower during registration. | float64 | 0.0% | — |
| `emp_length` | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year... | object | 6.1% | — |
| `emp_title` | The job title supplied by the Borrower when applying for the loan.* | object | 6.1% | — |
| `home_ownership` | The home ownership status provided by the borrower during registration. Our values are: RENT, OWN... | object | 0.0% | — |
| `verification_status` | Indicates if income was verified by LC, not verified, or if the income source was verified | object | 0.0% | — |
| `zip_code` | The first 3 numbers of the zip code provided by the borrower in the loan application. | object | 0.0% | — |

## 3. Credit History (Core)

| Column Name | Definition | Type | % Missing | Notes |
|---|---|---|---|---|
| `delinq_2yrs` | The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the p... | float64 | 0.0% | — |
| `dti` | A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations... | float64 | 0.0% | — |
| `earliest_cr_line` | The month the borrower's earliest reported credit line was opened | object | 0.0% | — |
| `fico_range_high` | The upper boundary of range the borrower’s FICO belongs to. | float64 | 0.0% | — |
| `fico_range_low` | The lower boundary of range the borrower’s FICO belongs to. | float64 | 0.0% | — |
| `inq_last_6mths` | The number of inquiries by creditors during the past 6 months. | float64 | 0.0% | — |
| `mths_since_last_delinq` | The number of months since the borrower's last delinquency. | float64 | 48.2% | — |
| `mths_since_last_record` | The number of months since the last public record. | float64 | 82.2% | Sparse |
| `open_acc` | The number of open credit lines in the borrower's credit file. | float64 | 0.0% | — |
| `pub_rec` | Number of derogatory public records | float64 | 0.0% | — |
| `revol_bal` | Total credit revolving balance | float64 | 0.0% | — |
| `revol_util` | Revolving line utilization rate, or the amount of credit the borrower is using relative to all av... | float64 | 0.0% | — |
| `total_acc` | The total number of credit lines currently in the borrower's credit file | float64 | 0.0% | — |

## 4. Credit Bureau — Extended Variables

| Column Name | Definition | Type | % Missing | Notes |
|---|---|---|---|---|
| `acc_now_delinq` | The number of accounts on which the borrower is now delinquent. | float64 | 0.0% | — |
| `acc_open_past_24mths` | Number of trades opened in past 24 months | float64 | 0.0% | — |
| `all_util` | Ratio of total current balance to total credit limit for all accounts | float64 | 78.6% | Sparse |
| `avg_cur_bal` | Average current balance of all accounts | float64 | 0.0% | — |
| `bc_open_to_buy` | Total open to buy on revolving bankcards. | float64 | 1.0% | — |
| `bc_util` | Ratio of total current balance to high credit/credit limit for all bankcard accounts. | float64 | 1.0% | — |
| `chargeoff_within_12_mths` | Number of charge-offs within 12 months | float64 | 0.0% | — |
| `delinq_amnt` | The past-due amount owed for the accounts on which the borrower is now delinquent. | float64 | 0.0% | — |
| `il_util` | Ratio of total current balance to high credit limit for installment accounts | float64 | 81.4% | Sparse |
| `inq_fi` | Number of personal finance inquiries | float64 | 78.6% | Sparse |
| `inq_last_12m` | Number of inquiries in past 12 months | float64 | 78.6% | Sparse |
| `max_bal_bc` | Maximum balance on all bankcard accounts | float64 | 78.6% | Sparse |
| `mo_sin_old_il_acct` | Months since oldest installment account opened | float64 | 2.8% | — |
| `mo_sin_old_rev_tl_op` | Months since oldest revolving account opened | float64 | 0.0% | — |
| `mo_sin_rcnt_rev_tl_op` | Months since most recent revolving account opened | float64 | 0.0% | — |
| `mo_sin_rcnt_tl` | Months since most recent account opened | float64 | 0.0% | — |
| `mort_acc` | Number of mortgage accounts. | float64 | 0.0% | — |
| `mths_since_recent_bc` | Months since most recent bankcard account opened. | float64 | 0.9% | — |
| `mths_since_recent_bc_dlq` | Months since most recent bankcard delinquency | float64 | 74.6% | Sparse |
| `mths_since_recent_inq` | Months since most recent inquiry | float64 | 10.8% | — |
| `mths_since_recent_revol_delinq` | Months since most recent revolving delinquency. | float64 | 64.1% | Sparse |
| `num_accts_ever_120_pd` | Number of accounts ever 120 or more days past due | float64 | 0.0% | — |
| `num_actv_bc_tl` | Number of currently active bankcard accounts | float64 | 0.0% | — |
| `num_actv_rev_tl` | Number of currently active revolving trades | float64 | 0.0% | — |
| `num_bc_sats` | Number of satisfactory bankcard accounts | float64 | 0.0% | — |
| `num_bc_tl` | Number of bankcard accounts | float64 | 0.0% | — |
| `num_il_tl` | Number of installment accounts | float64 | 0.0% | — |
| `num_op_rev_tl` | Number of open revolving accounts | float64 | 0.0% | — |
| `num_rev_accts` | Number of revolving accounts | float64 | 0.0% | — |
| `num_rev_tl_bal_gt_0` | Number of revolving trades with balance >0 | float64 | 0.0% | — |
| `num_sats` | Number of satisfactory accounts | float64 | 0.0% | — |
| `num_tl_120dpd_2m` | Number of accounts currently 120 days past due (updated in past 2 months) | float64 | 5.2% | — |
| `num_tl_30dpd` | Number of accounts currently 30 days past due (updated in past 2 months) | float64 | 0.0% | — |
| `num_tl_90g_dpd_24m` | Number of accounts 90 or more days past due in last 24 months | float64 | 0.0% | — |
| `num_tl_op_past_12m` | Number of accounts opened in past 12 months | float64 | 0.0% | — |
| `open_acc_6m` | Number of open accounts opened in past 6 months | float64 | 78.6% | Sparse |
| `open_act_il` | Number of currently active installment accounts | float64 | 78.6% | Sparse |
| `open_il_12m` | Number of installment accounts opened in past 12 months | float64 | 78.6% | Sparse |
| `open_il_24m` | Number of installment accounts opened in past 24 months | float64 | 78.6% | Sparse |
| `open_rv_12m` | Number of revolving accounts opened in past 12 months | float64 | 78.6% | Sparse |
| `open_rv_24m` | Number of revolving accounts opened in past 24 months | float64 | 78.6% | Sparse |
| `pct_tl_nvr_dlq` | Percent of trades never delinquent | float64 | 0.0% | — |
| `percent_bc_gt_75` | Percentage of bankcard accounts with utilization greater than 75% | float64 | 1.1% | — |
| `pub_rec_bankruptcies` | Number of public record bankruptcies | float64 | 0.0% | — |
| `tax_liens` | Number of tax liens | float64 | 0.0% | — |
| `tot_coll_amt` | Total collection amounts ever owed | float64 | 0.0% | — |
| `tot_cur_bal` | Total current balance of all accounts | float64 | 0.0% | — |
| `tot_hi_cred_lim` | Total high credit/credit limit | float64 | 0.0% | — |
| `total_bal_ex_mort` | Total credit balance excluding mortgage | float64 | 0.0% | — |
| `total_bal_il` | Total current balance of all installment accounts | float64 | 78.6% | Sparse |
| `total_bc_limit` | Total bankcard high credit/credit limit | float64 | 0.0% | — |
| `total_cu_tl` | Number of credit union trades | float64 | 78.6% | Sparse |
| `total_il_high_credit_limit` | Total installment high credit/credit limit | float64 | 0.0% | — |
| `total_rev_hi_lim` | Total high credit/credit limit for all revolving accounts | float64 | 0.0% | — |

## 5. Loan Performance (Post-Origination)

| Column Name | Definition | Type | % Missing | Notes |
|---|---|---|---|---|
| `collection_recovery_fee` | post charge off collection fee | float64 | 0.0% | ⚠️ Leakage |
| `collections_12_mths_ex_med` | Number of collections in 12 months excluding medical collections | float64 | 0.0% | — |
| `last_credit_pull_d` | The most recent month LC pulled credit for this loan | object | 0.0% | ⚠️ Leakage |
| `last_fico_range_high` | The last upper boundary of range the borrower’s FICO belongs to pulled. | float64 | 0.0% | ⚠️ Leakage |
| `last_fico_range_low` | The last lower boundary of range the borrower’s FICO belongs to pulled. | float64 | 0.0% | ⚠️ Leakage |
| `last_pymnt_amnt` | Last total payment amount received | float64 | 0.0% | ⚠️ Leakage |
| `last_pymnt_d` | Last month payment was received | object | 0.1% | ⚠️ Leakage |
| `loan_status` | Current status of the loan | object | 0.0% | ⚠️ Leakage |
| `mths_since_last_major_derog` | Months since most recent 90-day or worse rating | float64 | 70.6% | Sparse |
| `next_pymnt_d` | Next scheduled payment date | object | 87.9% | Sparse | ⚠️ Leakage |
| `out_prncp` | Remaining outstanding principal for total amount funded | float64 | 0.0% | ⚠️ Leakage |
| `out_prncp_inv` | Remaining outstanding principal for portion of total amount funded by investors | float64 | 0.0% | ⚠️ Leakage |
| `recoveries` | post charge off gross recovery | float64 | 0.0% | ⚠️ Leakage |
| `total_pymnt` | Payments received to date for total amount funded | float64 | 0.0% | ⚠️ Leakage |
| `total_pymnt_inv` | Payments received to date for portion of total amount funded by investors | float64 | 0.0% | ⚠️ Leakage |
| `total_rec_int` | Interest received to date | float64 | 0.0% | ⚠️ Leakage |
| `total_rec_late_fee` | Late fees received to date | float64 | 0.0% | ⚠️ Leakage |
| `total_rec_prncp` | Principal received to date | float64 | 0.0% | ⚠️ Leakage |

## 6. Hardship & Settlement Program

| Column Name | Definition | Type | % Missing | Notes |
|---|---|---|---|---|
| `debt_settlement_flag` | Indicates whether a loan has a settlement flag (Y/N) | object | 0.0% | — |
| `debt_settlement_flag_date` | Date when debt settlement flag was set | object | 97.1% | Sparse |
| `deferral_term` | Number of months deferral granted to borrower | float64 | 99.2% | Sparse |
| `hardship_amount` | The original loan amount after the borrower requested hardship | float64 | 99.2% | Sparse |
| `hardship_dpd` | Days past due borrower was at start of hardship program | float64 | 99.2% | Sparse |
| `hardship_end_date` | End date of hardship program | object | 99.2% | Sparse |
| `hardship_flag` | Indicates whether a loan is currently in a hardship program (Y/N) | object | 0.0% | — |
| `hardship_last_payment_amount` | Last payment amount received before hardship status reported | float64 | 99.2% | Sparse |
| `hardship_length` | Length of hardship plan in months | float64 | 99.2% | Sparse |
| `hardship_loan_status` | Status of loan after hardship completion (e.g., PAID_IN_FULL, CURRENT, CHARGED_OFF, DEFAULTED) | object | 99.2% | Sparse |
| `hardship_payoff_balance_amount` | Remaining balance at end of hardship program | float64 | 99.2% | Sparse |
| `hardship_reason` | Reason for requesting hardship program (e.g., BUSINESS_DISRUPTION, DEATH_OF_FAMILY_MEMBER, DISABI... | object | 99.2% | Sparse |
| `hardship_start_date` | Start date of hardship program | object | 99.2% | Sparse |
| `hardship_status` | Status of hardship plan (e.g., ACTIVE, COMPLETED, DEFAULTED, ENDED, PENDING) | object | 99.2% | Sparse |
| `hardship_type` | Type of hardship program (e.g., DEFERMENT, FORBEARANCE, MODIFIED_DUE_DATE, TEMPORARY_FORBEARANCE) | object | 99.2% | Sparse |
| `orig_projected_additional_accrued_interest` | Original projected additional accrued interest at start of hardship program | float64 | 99.4% | Sparse |
| `payment_plan_start_date` | Start date of payment plan if a payment plan is active | object | 99.2% | Sparse |
| `settlement_amount` | Amount paid towards settlement | float64 | 97.1% | Sparse |
| `settlement_date` | Date when settlement agreement was completed | object | 97.1% | Sparse |
| `settlement_percentage` | Percentage of principal amount paid towards settlement | float64 | 97.1% | Sparse |
| `settlement_status` | Status of settlement agreement (e.g., SETTLEMENT_OFFER_ACCEPTED, SETTLEMENT_OFFER_PENDING, SETTLE... | object | 97.1% | Sparse |
| `settlement_term` | Number of months for settlement payment plan | float64 | 97.1% | Sparse |

## 7. Secondary Applicant (Joint Applications)

| Column Name | Definition | Type | % Missing | Notes |
|---|---|---|---|---|
| `annual_inc_joint` | The annual income provided by the co-borrower during registration for joint applications | float64 | 99.5% | Sparse |
| `dti_joint` | The debt to income ratio of the co-borrower in a joint application | float64 | 99.5% | Sparse |
| `revol_bal_joint` | Total revolving balance on joint applicant accounts | float64 | 100.0% | Sparse | Empty |
| `sec_app_chargeoff_within_12_mths` | Number of charge-offs within 12 months for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_collections_12_mths_ex_med` | Number of collections in 12 months (excluding medical) for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_earliest_cr_line` | Earliest credit line opened for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_fico_range_high` | Secondary applicant FICO score range upper boundary | float64 | 100.0% | Sparse | Empty |
| `sec_app_fico_range_low` | Secondary applicant FICO score range lower boundary | float64 | 100.0% | Sparse | Empty |
| `sec_app_inq_last_6mths` | Number of inquiries in past 6 months for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_mort_acc` | Number of mortgage accounts for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_mths_since_last_major_derog` | Months since last major derogatory mark for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_num_rev_accts` | Number of revolving accounts for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_open_acc` | Number of open accounts for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_open_act_il` | Number of active installment accounts for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `sec_app_revol_util` | Revolving utilization rate for secondary applicant | float64 | 100.0% | Sparse | Empty |
| `verification_status_joint` | The verification status of the co-borrower's income in a joint application | object | 99.5% | Sparse |

## 8. Administrative/System

| Column Name | Definition | Type | % Missing | Notes |
|---|---|---|---|---|
| `application_type` | Indicates whether the application is an individual or joint application | object | 0.0% | — |
| `disbursement_method` | Method of fund disbursement (CASH or DIRECT_PAY) | object | 0.0% | — |
| `id` | A unique LC assigned ID for the loan listing. | int64 | 0.0% | — |
| `member_id` | A unique LC assigned Id for the borrower member. | float64 | 100.0% | Sparse | Empty |
| `policy_code` | publicly available policy_code=1
new products not publicly available policy_code=2 | float64 | 0.0% | — |

---

## Columns to Drop (Completely Empty)

The following columns are 100% missing in the first 100,000 rows and should be dropped:

- `member_id`
- `revol_bal_joint`
- `sec_app_chargeoff_within_12_mths`
- `sec_app_collections_12_mths_ex_med`
- `sec_app_earliest_cr_line`
- `sec_app_fico_range_high`
- `sec_app_fico_range_low`
- `sec_app_inq_last_6mths`
- `sec_app_mort_acc`
- `sec_app_mths_since_last_major_derog`
- `sec_app_num_rev_accts`
- `sec_app_open_acc`
- `sec_app_open_act_il`
- `sec_app_revol_util`

## Variables Added After Official Dictionary

The following columns appear in the CSV but are not defined in LCDataDictionary.xlsx. Definitions are inferred from LendingClub's production documentation:

### Administrative

| Column Name | Definition | Type | % Missing |
|---|---|---|---|
| `disbursement_method` | Method of fund disbursement (CASH or DIRECT_PAY) | object | 0.0% |

### Borrower Status

| Column Name | Definition | Type | % Missing |
|---|---|---|---|
| `verification_status` | Indicates if income was verified by LC, not verified, or if the income source was verified | object | 0.0% |

### Credit Bureau Enhanced Variables

| Column Name | Definition | Type | % Missing |
|---|---|---|---|
| `acc_open_past_24mths` | Number of trades opened in past 24 months | float64 | 0.0% |
| `all_util` | Ratio of total current balance to total credit limit for all accounts | float64 | 78.6% |
| `il_util` | Ratio of total current balance to high credit limit for installment accounts | float64 | 81.4% |
| `inq_fi` | Number of personal finance inquiries | float64 | 78.6% |
| `inq_last_12m` | Number of inquiries in past 12 months | float64 | 78.6% |
| `max_bal_bc` | Maximum balance on all bankcard accounts | float64 | 78.6% |
| `mo_sin_old_il_acct` | Months since oldest installment account opened | float64 | 2.8% |
| `mths_since_rcnt_il` | Months since most recent installment account opened | float64 | 79.2% |
| `mths_since_recent_bc_dlq` | Months since most recent bankcard delinquency | float64 | 74.6% |
| `mths_since_recent_inq` | Months since most recent inquiry | float64 | 10.8% |
| `open_acc_6m` | Number of open accounts opened in past 6 months | float64 | 78.6% |
| `open_act_il` | Number of currently active installment accounts | float64 | 78.6% |
| `open_il_12m` | Number of installment accounts opened in past 12 months | float64 | 78.6% |
| `open_il_24m` | Number of installment accounts opened in past 24 months | float64 | 78.6% |
| `open_rv_12m` | Number of revolving accounts opened in past 12 months | float64 | 78.6% |
| `open_rv_24m` | Number of revolving accounts opened in past 24 months | float64 | 78.6% |
| `percent_bc_gt_75` | Percentage of bankcard accounts with utilization greater than 75% | float64 | 1.1% |
| `total_bal_il` | Total current balance of all installment accounts | float64 | 78.6% |
| `total_cu_tl` | Number of credit union trades | float64 | 78.6% |
| `total_rev_hi_lim` | Total high credit/credit limit for all revolving accounts | float64 | 0.0% |

### Hardship & Settlement Program

| Column Name | Definition | Type | % Missing |
|---|---|---|---|
| `debt_settlement_flag` | Indicates whether a loan has a settlement flag (Y/N) | object | 0.0% |
| `debt_settlement_flag_date` | Date when debt settlement flag was set | object | 97.1% |
| `deferral_term` | Number of months deferral granted to borrower | float64 | 99.2% |
| `hardship_amount` | The original loan amount after the borrower requested hardship | float64 | 99.2% |
| `hardship_dpd` | Days past due borrower was at start of hardship program | float64 | 99.2% |
| `hardship_end_date` | End date of hardship program | object | 99.2% |
| `hardship_flag` | Indicates whether a loan is currently in a hardship program (Y/N) | object | 0.0% |
| `hardship_last_payment_amount` | Last payment amount received before hardship status reported | float64 | 99.2% |
| `hardship_length` | Length of hardship plan in months | float64 | 99.2% |
| `hardship_loan_status` | Status of loan after hardship completion (e.g., PAID_IN_FULL, CURRENT, CHARGED_OFF, DEFAULTED) | object | 99.2% |
| `hardship_payoff_balance_amount` | Remaining balance at end of hardship program | float64 | 99.2% |
| `hardship_reason` | Reason for requesting hardship program (e.g., BUSINESS_DISRUPTION, DEATH_OF_FAMILY_MEMBER, DISABI... | object | 99.2% |
| `hardship_start_date` | Start date of hardship program | object | 99.2% |
| `hardship_status` | Status of hardship plan (e.g., ACTIVE, COMPLETED, DEFAULTED, ENDED, PENDING) | object | 99.2% |
| `hardship_type` | Type of hardship program (e.g., DEFERMENT, FORBEARANCE, MODIFIED_DUE_DATE, TEMPORARY_FORBEARANCE) | object | 99.2% |
| `orig_projected_additional_accrued_interest` | Original projected additional accrued interest at start of hardship program | float64 | 99.4% |
| `payment_plan_start_date` | Start date of payment plan if a payment plan is active | object | 99.2% |
| `settlement_amount` | Amount paid towards settlement | float64 | 97.1% |
| `settlement_date` | Date when settlement agreement was completed | object | 97.1% |
| `settlement_percentage` | Percentage of principal amount paid towards settlement | float64 | 97.1% |
| `settlement_status` | Status of settlement agreement (e.g., SETTLEMENT_OFFER_ACCEPTED, SETTLEMENT_OFFER_PENDING, SETTLE... | object | 97.1% |
| `settlement_term` | Number of months for settlement payment plan | float64 | 97.1% |

### Joint Application & Secondary Applicant

| Column Name | Definition | Type | % Missing |
|---|---|---|---|
| `annual_inc_joint` | The annual income provided by the co-borrower during registration for joint applications | float64 | 99.5% |
| `application_type` | Indicates whether the application is an individual or joint application | object | 0.0% |
| `dti_joint` | The debt to income ratio of the co-borrower in a joint application | float64 | 99.5% |
| `revol_bal_joint` | Total revolving balance on joint applicant accounts | float64 | 100.0% |
| `sec_app_chargeoff_within_12_mths` | Number of charge-offs within 12 months for secondary applicant | float64 | 100.0% |
| `sec_app_collections_12_mths_ex_med` | Number of collections in 12 months (excluding medical) for secondary applicant | float64 | 100.0% |
| `sec_app_earliest_cr_line` | Earliest credit line opened for secondary applicant | float64 | 100.0% |
| `sec_app_fico_range_high` | Secondary applicant FICO score range upper boundary | float64 | 100.0% |
| `sec_app_fico_range_low` | Secondary applicant FICO score range lower boundary | float64 | 100.0% |
| `sec_app_inq_last_6mths` | Number of inquiries in past 6 months for secondary applicant | float64 | 100.0% |
| `sec_app_mort_acc` | Number of mortgage accounts for secondary applicant | float64 | 100.0% |
| `sec_app_mths_since_last_major_derog` | Months since last major derogatory mark for secondary applicant | float64 | 100.0% |
| `sec_app_num_rev_accts` | Number of revolving accounts for secondary applicant | float64 | 100.0% |
| `sec_app_open_acc` | Number of open accounts for secondary applicant | float64 | 100.0% |
| `sec_app_open_act_il` | Number of active installment accounts for secondary applicant | float64 | 100.0% |
| `sec_app_revol_util` | Revolving utilization rate for secondary applicant | float64 | 100.0% |
| `verification_status_joint` | The verification status of the co-borrower's income in a joint application | object | 99.5% |

---

## Data Quality & Missingness Analysis

### High Missingness Patterns (>50%)

| Variable Group | Reason | Impact |
|---|---|---|
| Joint Application Variables (~99%) | Single-applicant loans dominate | Exclude from models unless specifically modeling joint applications |
| Secondary Applicant Variables (100%) | No joint applications in this dataset | Drop entirely |
| Recent Bureau Variables (~78%) | Only available for loans originated after ~2013 Q1 | Create separate model tracks or use for post-2013 subset |
| Hardship Program Variables (~99%) | Most loans did not enter hardship | Post-origination; exclude from PD models |
| Settlement Program Variables (~97%) | Most loans did not settle | Post-origination; exclude from PD models |

### Data Quality Issues by Type

**Text/Object Fields Requiring Parsing:**
- `issue_d`, `earliest_cr_line`, `last_pymnt_d`, `next_pymnt_d`, `last_credit_pull_d`: Convert to datetime
- `term`: Extract numeric value (36 or 60 months)
- `emp_length`: Handle categorical values (0-10 years, or convert to numeric)
- `desc`: 99.99% missing — drop

**Ratio Fields:**
- Utilization ratios (`revol_util`, `bc_util`, `il_util`, `all_util`): May contain values >100% or <0%
- DTI (`dti`, `dti_joint`): Verify range is 0-1 or 0-100

**Missing Indicators:**
- Some variables (e.g., `mths_since_last_delinq`) have natural missingness when the event hasn't occurred
- Create binary indicators for imputation: `has_delinq_history`, `has_public_record`, etc.

### Recommended Variable Selection

**For PD (Probability of Default) Models:**
1. Core credit variables (Section 3)
2. Extended bureau variables (Section 4)
3. Borrower demographics (Section 2)
4. Loan structure (Section 1: grade, sub_grade, purpose, term, int_rate)

**Exclude:**
- All post-origination variables (Section 5)
- All hardship/settlement variables (Section 6)
- Joint applicant variables (unless specifically modeling co-borrower credit)

**For LGD (Loss Given Default) / Recovery Models:**
1. Loan amount and structure (loan_amnt, funded_amnt, grade, sub_grade)
2. Collateral indicators (home_ownership, mort_acc)
3. Recovery tracking (total_rec_prncp, total_rec_int, recoveries, recovery_fee)
4. Payment history (total_pymnt, last_pymnt_amnt)

**Dangerous Leakage Variables (MUST EXCLUDE from prediction models):**
- `loan_status` — target variable
- `out_prncp`, `out_prncp_inv` — outstanding principal (outcome-dependent)
- `total_pymnt*`, `total_rec_*` — payment and recovery amounts
- `last_pymnt_d`, `last_pymnt_amnt` — post-origination payment info
- `last_fico_range_*` — updated during loan lifecycle
- `last_credit_pull_d` — date of pull during servicing
- All `hardship_*` and `settlement_*` variables — post-origination events
