#!/bin/bash

mapfile -d $'\0' array < <(find .. -type d -name "focus_on" -print0)
yesterday_date=$(date -d "yesterday" +%Y%m%d)

for dir in "${array[@]}"
do
    to_be_replaced=$(basename "$dir")
    parent_dir=$(dirname "$dir")
    new_folder_name="$yesterday_date"
    mv "$dir" "${parent_dir}/${new_folder_name}"
    echo "$dir has been move to $parent_dir/$new_folder_name"
done
