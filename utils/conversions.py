def convert_pixels_to_meters(pixels, reference_meters, reference_pixels):
    return (pixels * reference_meters) / reference_pixels

def convert_meters_to_pixels(meters, reference_pixels, reference_meters):
    return (meters * reference_pixels) / reference_meters