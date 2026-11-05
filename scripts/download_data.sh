#!/usr/bin/env bash
# Downloads the Loghub HDFS_v1 dataset (Xu et al., SOSP 2009) into data/HDFS_v1/
# and verifies the MD5 checksum published on Zenodo.
#
# Dataset:  HDFS_v1, Loghub collection, https://zenodo.org/records/8196385
# License:  The datasets are freely available for research or academic work. For any
#           usage or distribution, refer to https://github.com/logpai/loghub and cite:
#             Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. Loghub: A Large
#             Collection of System Log Datasets for AI-driven Log Analytics. In ISSRE, 2023.
#           The original data come from:
#             Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael I. Jordan. Detecting
#             Large-Scale System Problems by Mining Console Logs. In SOSP, 2009.
#
# Usage: scripts/download_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/HDFS_v1"
ZIP="$ROOT/data/HDFS_v1.zip"
URL="https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
MD5="76a24b4d9a6164d543fb275f89773260"

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
}

mkdir -p "$ROOT/data"
if [[ -f "$DEST/HDFS.log" && -f "$DEST/preprocessed/anomaly_label.csv" ]]; then
  echo "HDFS_v1 already present in $DEST"
  exit 0
fi

if [[ ! -f "$ZIP" ]] || [[ "$(md5_of "$ZIP")" != "$MD5" ]]; then
  echo "==> downloading HDFS_v1.zip (187 MB) from Zenodo"
  curl -fsSL --retry 3 -o "$ZIP" "$URL"
fi
actual="$(md5_of "$ZIP")"
if [[ "$actual" != "$MD5" ]]; then
  echo "error: MD5 mismatch for HDFS_v1.zip (expected $MD5, got $actual)" >&2
  exit 1
fi

echo "==> extracting to $DEST"
mkdir -p "$DEST"
unzip -q -o "$ZIP" -d "$DEST"
ls -la "$DEST"
