## eKYC Logs Analysis
# eKYC Process
# Definition:
Electronic Know Your Customer (eKYC) is the digital process used to verify a customer's identity remotely. It typically involves authenticating the individual using biometric details linked to their UID (Aadhaar) number.

# Scenarios:

- New LPG connection requests
- Existing LPG consumers updating their details

# Typical Conditions:

- If done by the distributor, it usually occurs during business hours (09:00 – 17:00).
- Expected completion time: Minimum 1 minute, average 2–3 minutes per transaction.

# Process Steps:

1) Consent approval from the customer.
2) Aadhaar number entry and selection of authentication mode:
   - Facial Recognition — LpgNext
   - Iris Recognition — LpgNext
   - Face Authentication — Hello BPCL app, eKYC Operator app

3) Biometric data capture
4) Biometric verification result
5) Transaction status update

# Data Fields
- eKYC Transaction ID:
  Unique, auto-incremented ID for each attempt.

- DistCode:
  Distributor code (unique per distributor).
  Sub-users: Codes ending with _01, _02, … _20

- CreatedBy:
 Indicates who initiated the transaction:
   - Distributor user: CC prefix (Customer Code)
   - Distributor sub-user: CC prefix + _01 … _20 suffix
   - eKYC operator: EO prefix
   - Consumer via Hello BPCL app: LPGID populated (BPC LPGIDs start with 1)

- InsertedOn:
  Timestamp when the transaction was initiated.

- eKYCStatus:
  Current state of the process:
  0 — Pending
  1 or 2 — Success
  3 — Failure

- AppID:
  Source system of initiation:
  1 — LpgNext Portal
  6 — Hello BPCL app
  7 — eKYC Operator app

- ConsumerID:
  Unique identifier assigned to each customer.

- UpdatedOn:
  Timestamp when the process was completed.

# Current Checks for Suspicious Activity
Per User:
 - Transactions per minute: Max 1
 - Transactions per hour: Max 75

Per Distributor:
 - Transactions per minute: Max 5
 - Transactions per hour: Max 150

# Potential Areas for Anomaly Detection
- Distributor behavior over time:
  Track transaction counts per distributor across time intervals to detect sudden spikes or irregular patterns.

- Activity at unusual hours:
  Identify eKYC attempts occurring outside standard business hours (09:00 – 17:00).

- Customer-specific irregularities:
  Monitor repeated or unusual activity patterns for the same customer over time.