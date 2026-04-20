#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

load_lab_env
ACCOUNT_ID="$(aws_account_id)"
BUDGETS_REGION="us-east-1"

: "${BUDGET_NAME:=${PROJECT_NAME}-monthly}"
require_env MONTHLY_BUDGET_LIMIT_USD BUDGET_EMAIL

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

BUDGET_FILE="${TMP_DIR}/budget.json"
NOTIFICATIONS_FILE="${TMP_DIR}/notifications.json"

cat >"${BUDGET_FILE}" <<EOF
{
  "BudgetName": "${BUDGET_NAME}",
  "BudgetLimit": {
    "Amount": "${MONTHLY_BUDGET_LIMIT_USD}",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST",
  "CostTypes": {
    "IncludeTax": true,
    "IncludeSubscription": true,
    "UseBlended": false,
    "IncludeRefund": false,
    "IncludeCredit": false,
    "IncludeUpfront": true,
    "IncludeRecurring": true,
    "IncludeOtherSubscription": true,
    "IncludeSupport": true,
    "IncludeDiscount": true,
    "UseAmortized": false
  }
}
EOF

cat >"${NOTIFICATIONS_FILE}" <<EOF
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "${BUDGET_EMAIL}"
      }
    ]
  },
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 100,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "${BUDGET_EMAIL}"
      }
    ]
  }
]
EOF

if aws --region "${BUDGETS_REGION}" budgets describe-budget --account-id "${ACCOUNT_ID}" --budget-name "${BUDGET_NAME}" >/dev/null 2>&1; then
  aws --region "${BUDGETS_REGION}" budgets update-budget \
    --account-id "${ACCOUNT_ID}" \
    --new-budget "file://${BUDGET_FILE}" >/dev/null
  echo "Updated budget: ${BUDGET_NAME}"
else
  aws --region "${BUDGETS_REGION}" budgets create-budget \
    --account-id "${ACCOUNT_ID}" \
    --budget "file://${BUDGET_FILE}" \
    --notifications-with-subscribers "file://${NOTIFICATIONS_FILE}" >/dev/null
  echo "Created budget: ${BUDGET_NAME}"
fi
