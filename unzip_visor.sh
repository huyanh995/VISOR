find ../GroundTruth-SparseAnnotations/rgb_frames -iname '*.zip' -exec sh -c 'unzip -o -d "${0%.*}" "$0"' '{}' ';'
