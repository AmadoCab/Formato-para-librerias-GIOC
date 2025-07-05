#!/usr/bin/env bash

# Variables
output_file="wofram-dump.wl"
directories=("libsCP" "libsJA" "QW")
extensions=("m" "wl")

# Output file cleaning
> "$output_file"

# Searching in each directory
for dir in "${directories[@]}"; do
  echo "Updating $dir..."
  git -C "$dir" pull --quiet

  for ext in "${extensions[@]}"; do
    find "$dir" -type f -name "*.$ext" | while read -r file; do
      # If it's a .wl file, include it directly
      if [[ "$file" == *.wl ]]; then
        echo -e "\n(* --- Begin: $file --- *)\n\n" >> "$output_file"
        cat "$file" >> "$output_file"
        echo -e "\n\n" >> "$output_file"
        echo -e "\n(* --- End: $file --- *)\n\n" >> "$output_file"

      # If it's a .m file, check if it looks like a Mathematica package
      elif [[ "$file" == *.m ]]; then
        if grep -Eq 'Begin\["`?Private`"\]' "$file"; then
          echo -e "\n(* --- Begin: $file --- *)\n\n" >> "$output_file"
          cat "$file" >> "$output_file"
          echo -e "\n\n" >> "$output_file"
          echo -e "\n(* --- End: $file --- *)\n\n" >> "$output_file"
        fi
      fi
    done
  done
done

echo "Combined files written to $output_file"
