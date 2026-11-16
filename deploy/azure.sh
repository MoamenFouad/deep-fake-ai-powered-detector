#!/usr/bin/env bash
# Deploy DeepShield to Azure Container Apps (works with an Azure for Students account).
#
# Run it in the Azure Cloud Shell (https://shell.azure.com, choose Bash):
#   curl -fsSL https://raw.githubusercontent.com/MoamenFouad/deep-fake-ai-powered-detector/main/deploy/azure.sh | bash
#
# Before running, make the Docker image public once: on GitHub open your profile,
# Packages, deep-fake-ai-powered-detector, Package settings, Change visibility, Public.
#
# What it creates (all in one resource group, so one command deletes everything):
#   * a Container Apps environment with logging turned off (no Log Analytics charges)
#   * the app: 2 vCPU / 4 GiB, scales to zero when idle, at most 1 copy running
#
# Settings can be changed with environment variables, for example:
#   REGION=northeurope bash azure.sh
set -euo pipefail

APP="${APP:-deepshield}"
RG="${RG:-deepshield-rg}"
ENVNAME="${ENVNAME:-deepshield-env}"
IMAGE="${IMAGE:-ghcr.io/moamenfouad/deep-fake-ai-powered-detector:latest}"
CPU="${CPU:-2.0}"
MEMORY="${MEMORY:-4Gi}"
# Student subscriptions only allow some regions; these are tried in order.
REGIONS="${REGION:-westeurope northeurope swedencentral francecentral uaenorth eastus eastus2 westus2 centralindia}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/5  Checking the Azure account"
az account show --query "{subscription:name, user:user.name}" -o table

say "2/5  Preparing the Azure CLI and resource providers (first time only, takes a minute)"
az extension add --name containerapp --upgrade --only-show-errors -y >/dev/null
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

say "3/5  Creating the environment"
LOCATION=""
EXISTING=$(az containerapp env show -n "$ENVNAME" -g "$RG" --query location -o tsv 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  LOCATION="$EXISTING"
  echo "Environment already exists in $LOCATION"
else
  for r in $REGIONS; do
    echo "Trying region $r ..."
    az group create -n "$RG" -l "$r" -o none 2>/dev/null || true
    if az containerapp env create -n "$ENVNAME" -g "$RG" -l "$r" --logs-destination none -o none 2>/tmp/ds_env_err; then
      LOCATION="$r"; break
    fi
    echo "  not available: $(tail -1 /tmp/ds_env_err | cut -c1-160)"
    az group delete -n "$RG" --yes --no-wait -o none 2>/dev/null || true
    sleep 5
  done
fi
[ -n "$LOCATION" ] || { echo "No allowed region worked. Set REGION=<a region your subscription allows> and run again."; exit 1; }

say "4/5  Deploying the app in $LOCATION (pulling a ~3 GB image, takes a few minutes)"
if az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  # A new revision suffix makes Azure pull the latest image even when the tag is unchanged.
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --cpu "$CPU" --memory "$MEMORY" \
    --min-replicas 0 --max-replicas 1 --revision-suffix "r$(date +%m%d%H%M%S)" -o none
else
  az containerapp create -n "$APP" -g "$RG" --environment "$ENVNAME" --image "$IMAGE" \
    --ingress external --target-port 7860 --cpu "$CPU" --memory "$MEMORY" \
    --min-replicas 0 --max-replicas 1 --env-vars OMP_NUM_THREADS=2 -o none
fi

URL="https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"

say "5/5  Waking the app up (the first start loads the models, up to 3 minutes)"
for i in $(seq 1 36); do
  if curl -fs -m 20 "$URL/api/health" | grep -q '"models_loaded":true'; then
    echo "Ready."; break
  fi
  sleep 5
done

say "DeepShield is live at: $URL"
cat <<EOF

It sleeps when nobody uses it, so the first visit after a pause takes 1 to 3 minutes
(if the browser gives up, reload the page).
Presentation day, keep it awake:   az containerapp update -n $APP -g $RG --min-replicas 1
Afterwards, let it sleep again:    az containerapp update -n $APP -g $RG --min-replicas 0
Delete everything:                 az group delete -n $RG --yes
EOF
