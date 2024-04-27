

"""Function that calculates the forest area based on the percentage of the forest in the image
Args:
    percentage (float): The percentage of the forest in the image (0-1)
"""
def calculate_forest_area(percentage):
    # Each pixel represents a 50*50cm area
    pixel_area = 0.25
    # Number of pixels in the image
    total_pixels = 2448 * 2448
    # Calculate the total area
    total_area = total_pixels * pixel_area
    # Calculate the forest area based on the percentage

    forest_area = total_area * percentage
    return forest_area


"""Function that calculates oxygen produced by the forest based on the forest area
Args:
    forest_area (float): The area of the forest in square meters
"""
def calculate_yearly_oxygen_production(forest_area):
    # Oxygen production in grams per square meter
    # 100kg per year per tree
    # 1 tree = 1m^2

    oxygen_per_sqm = 100
    # Calculate the total oxygen production
    total_oxygen = forest_area * oxygen_per_sqm
    return total_oxygen

"""Function that calculates carbon captured by the forest based on the forest area
Args:
    forest_area (float): The area of the forest in square meters
"""
def calculate_yearly_carbon_capture(forest_area):
    # 22kg per year per tree
    # 1 tree = 1m^2

    carbon_per_sm = 22
    # Calculate the total oxygen production
    total_carb = forest_area * carbon_per_sm
    return total_carb

