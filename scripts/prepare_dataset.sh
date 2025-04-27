#!/bin/bash

# filepath: /home/lore/workspace/github.com/LoreviQ/Baxus-Whisky-Goggles/scripts/prepare_dataset.sh

# Define source and target directories
SOURCE_DIR="./data/source_images"
AUGMENTED_DIR="./data/augmented_images"
TARGET_DIR="./data/training_images"

# Create the target directory if it doesn't exist
echo "Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Process source images
echo "Processing source images from $SOURCE_DIR..."
for img_path in "$SOURCE_DIR"/*.png; do
    if [ -f "$img_path" ]; then
        filename=$(basename "$img_path")
        image_id="${filename%.png}"

        # Create subdirectory for the image_id
        mkdir -p "$TARGET_DIR/$image_id"

        # Copy and rename the source image
        cp "$img_path" "$TARGET_DIR/$image_id/source.png"
        # echo "Copied $img_path to $TARGET_DIR/$image_id/source.png"
    fi
done
echo "Finished processing source images."

# Process augmented images
echo "Processing augmented images from $AUGMENTED_DIR..."
for img_path in "$AUGMENTED_DIR"/*.png; do
     if [ -f "$img_path" ]; then
        filename=$(basename "$img_path")
        # Extract image_id and number using parameter expansion
        base_name="${filename%.png}"
        image_id="${base_name%_*}"
        number="${base_name##*_}"

        # Check if the target directory exists (created by source image processing)
        if [ -d "$TARGET_DIR/$image_id" ]; then
            # Copy and rename the augmented image
            cp "$img_path" "$TARGET_DIR/$image_id/augmented_${number}.png"
            # echo "Copied $img_path to $TARGET_DIR/$image_id/augmented_${number}.png"
        else
            echo "Warning: Target directory $TARGET_DIR/$image_id does not exist for augmented image $filename. Skipping."
        fi
    fi
done
echo "Finished processing augmented images."

echo "Dataset preparation complete. Output in $TARGET_DIR"