#!/bin/bash

py=python3

#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 <folder_with_numpy_files> <extension> <pipeline_file>"
  exit 1
fi

FOLDER_WITH_NUMPY_FILES="$1"
EXTENSION="$2"
export PIPELINE_FILE="$3"

export SCRIPT_PATH="analysis/visualization/generate_video_from_expmaps.py"

# Iterate over all .npy files in the given folder
# for FILE in "$FOLDER_WITH_NUMPY_FILES"/*.${EXTENSION}; do
#   echo "Processing file $FILE..."
#   $py "$SCRIPT_PATH" \
#       --features_file "$FILE" \
#       --pipeline_file "$PIPELINE_FILE" \
#       --generate_bvh
#       # Add other optional arguments as needed
# done

find "$FOLDER_WITH_NUMPY_FILES" -name "*.${EXTENSION}" | parallel -j 6 'echo "Processing file {}..."; python "$SCRIPT_PATH" --features_file "{}" --pipeline_file "$PIPELINE_FILE" --generate_bvh'

echo "All files processed."
