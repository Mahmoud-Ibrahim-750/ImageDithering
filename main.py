from dithering_methods import *
import time

# Usage Examples:
# Start measuring the execution time
start_time = time.time()

# Call the function
threshold_dithering('input.jpg', 'output_threshold.jpg', 128)

# End measuring the execution time
end_time = time.time()

# Print the execution time in seconds
print("Threshold execution time: ", end_time - start_time, "seconds")

start_time = time.time()
improved_threshold_dithering('input.jpg', 'output_threshold_improved.jpg', 128)
end_time = time.time()
print("Improved Threshold execution time: ", end_time - start_time, "seconds")

print()  # separation line

start_time = time.time()
floyd_steinberg('input.jpg', 'output_floyd.jpg')
end_time = time.time()
print("Floyd-Steinberg execution time: ", end_time - start_time, "seconds")

print()  # separation line

start_time = time.time()
ordered_dithering('input.jpg', 'output_ordered.jpg')
end_time = time.time()
print("Ordered execution time: ", end_time - start_time, "seconds")

start_time = time.time()
improved_ordered_dithering('input.jpg', 'output_ordered_improved.jpg')
end_time = time.time()
print("Improved Ordered execution time: ", end_time - start_time, "seconds")

print()  # separation line

# Try using different sized Bayer matrices (2 and 4)
start_time = time.time()
# Generate 2x2 Bayer matrix
matrix = bayer_matrix(2)
# Apply pattern dithering to input image
pattern_dithering('input.jpg', 'output_pattern_2shades.jpg', matrix)
end_time = time.time()
print("Pattern (2 shades of gray) execution time: ", end_time - start_time, "seconds")

start_time = time.time()
matrix = bayer_matrix(4)
pattern_dithering('input.jpg', 'output_pattern_4shades.jpg', matrix)
end_time = time.time()
print("Pattern (4 shades of gray) execution time: ", end_time - start_time, "seconds")


print()  # separation line


# Try using an improved version with same matrices (2 and 4)
start_time = time.time()
matrix = improved_bayer_matrix(2)
improved_pattern_dithering('input.jpg', 'output_pattern_2shades_improved.jpg', matrix)
end_time = time.time()
print("Pattern (2 shades) - improved execution time: ", end_time - start_time, "seconds")

start_time = time.time()
matrix2 = improved_bayer_matrix(4)
improved_pattern_dithering('input.jpg', 'output_pattern_4shades_improved.jpg', matrix2)
end_time = time.time()
print("Pattern (4 shades) - improved execution time: ", end_time - start_time, "seconds")
