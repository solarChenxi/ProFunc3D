#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MASK3D_DIR="${REPO_ROOT}/third_party/Mask3D"
PATCH_FILE="${REPO_ROOT}/third_party/mask3d_f.patch"
MASK3D_COMMIT="11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5"

if [[ ! -d "${MASK3D_DIR}/.git" ]]; then
  git clone https://github.com/JonasSchult/Mask3D.git "${MASK3D_DIR}"
fi

if ! git -C "${MASK3D_DIR}" cat-file -e "${MASK3D_COMMIT}^{commit}"; then
  git -C "${MASK3D_DIR}" fetch origin "${MASK3D_COMMIT}"
fi

git -C "${MASK3D_DIR}" checkout "${MASK3D_COMMIT}"

if git -C "${MASK3D_DIR}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  echo "Mask3D-F patch is already applied."
else
  git -C "${MASK3D_DIR}" apply --check "${PATCH_FILE}"
  git -C "${MASK3D_DIR}" apply "${PATCH_FILE}"
  echo "Applied Mask3D-F patch."
fi

echo "Mask3D-F backend ready at ${MASK3D_DIR}"
