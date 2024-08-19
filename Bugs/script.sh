#!/bin/bash

# This script navigates through all subdirectories, temporarily moves 'original_code_snippet.py' if it exists,
# runs the 'pipreqs' command to generate 'requirements.txt' files based on Python imports, and restores the file.

# Find all directories in the current directory
for dir in */ ; do
    # Navigate into the directory
    cd "$dir"

    # Check if 'original_code_snippet.py' exists and temporarily move it
    if [ -f "original_code_snippet.py" ]; then
        echo "'original_code_snippet.py' found in $dir, temporarily moving it."
        mv "original_code_snippet.py" "original_code_snippet.py.bak"
        file_moved=true
    else
        file_moved=false
    fi

    # Run pipreqs to generate requirements.txt for Python projects
    echo "Running pipreqs in $dir"
    pipreqs --force .

    # Restore 'original_code_snippet.py' if it was moved
    if [ "$file_moved" = true ]; then
        mv "original_code_snippet.py.bak" "original_code_snippet.py"
        echo "'original_code_snippet.py' has been restored in $dir."
    fi

    # Navigate back to the parent directory
    cd ..

done

echo "Operation completed."