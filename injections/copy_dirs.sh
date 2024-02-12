#!/bin/bash

DIR1="./outdir_TaylorF2/"
DIR2="/home/thibeau.wouters/public_html/jim_injections/tests_TaylorF2_19_01_2024/"

# Check if DIR1 exists
if [ ! -d "$DIR1" ]; then
  echo "Error: $DIR1 does not exist."
  exit 1
fi

# Check if DIR2 exists
if [ ! -d "$DIR2" ]; then
  echo "Error: $DIR2 does not exist."
  exit 1
fi

# Loop through subdirectories in DIR1
for subdirectory in "$DIR1"/*/; do
  # Extract the subdirectory name
  subdirectory_name=$(basename "$subdirectory")

  # Check if the same subdirectory exists in DIR2
  if [ ! -d "$DIR2/$subdirectory_name" ]; then
    # If not, copy the subdirectory to DIR2
    cp -r "$subdirectory" "$DIR2"
    echo "Copied: $subdirectory_name"
  else
    echo "Skipped: $subdirectory_name (Already exists in DIR2)"
  fi
done

echo "Done!"

