#!/bin/bash
# Utility script to find the nvidia_icd.json file and configure Vulkan for SAPIEN headless rendering.
# Usage:
#   source scripts/prepare_sapien.sh

echo "🔍 Searching for nvidia_icd.json for Vulkan..."

# Common directories for Vulkan ICDs
SEARCH_DIRS=("/usr/share/vulkan/icd.d" "/etc/vulkan/icd.d" "/usr/local/share/vulkan/icd.d")
ICD_PATH=""

for dir in "${SEARCH_DIRS[@]}"; do
    if [ -f "$dir/nvidia_icd.json" ]; then
        ICD_PATH="$dir/nvidia_icd.json"
        break
    fi
done

# Fallback to a system-wide search if not found in common directories
if [ -z "$ICD_PATH" ]; then
    echo "Not found in common directories. Running system-wide find (this may take a moment)..."
    ICD_PATH=$(find /usr -name "nvidia_icd.json" -type f 2>/dev/null | head -n 1)
fi

if [ -z "$ICD_PATH" ]; then
    echo "❌ Error: nvidia_icd.json not found."
    echo "Make sure NVIDIA drivers and Vulkan (libvulkan1) are properly installed."
    # Don't exit if sourced, just return
    return 1 2>/dev/null || exit 1
fi

echo "✅ Found NVIDIA ICD at: $ICD_PATH"

export VK_ICD_FILENAMES="$ICD_PATH"
echo "✅ Exported VK_ICD_FILENAMES=$VK_ICD_FILENAMES"

# Check if the script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo ""
    echo "⚠️  NOTE: You ran this script as an executable. The export will only apply to this script's subshell."
    echo "To apply the export to your current terminal session, run it with 'source':"
    echo ""
    echo "    source $0"
    echo ""
fi
