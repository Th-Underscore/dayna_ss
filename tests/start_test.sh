#!/bin/sh

# Ensure the router is running (once per session)
pgrep -f model_router.py || (
  nohup python3 tools/model_router.py --port 5007 --host 127.0.0.1 > /tmp/model_router.log 2>&1 &
  disown
  echo "Router started"
)

# Run the soak test
python3 long_horizon_soak.py \
  --spreadsheet spreadsheets/greek_mythology_retelling.json \
  --turns 8 \
  --schema-type 2 \
  --message-mode rolling \
  --retrieval-placement system \
  --rolling-summaries 20 \
  --max-scene-messages 8 \
  --cadence-profile compressed \
  --embed-device cpu \
  --guide-model mistral/ministral-3b-latest \
  --judge-model mistral/ministral-3b-latest \
  --auditor-model mistral/ministral-3b-latest \
  --judge-every 5 \
  --overview-every 8 \
  --audit-every 6 \
  --seed 104 \
  --cloud-endpoint http://localhost:5007 \
  --cloud-provider mistral
