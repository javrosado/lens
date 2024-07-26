import math
import cv2
import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import SymLogNorm

def find_template_location(original, template): #finds template
    result = cv2.matchTemplate(original, template, cv2.TM_CCOEFF_NORMED)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc



def calcScale(original_image, template, original_image_width, original_image_height):
        
    original_image_resolutionx = original_image.shape[1]
    original_image_resolutiony = original_image.shape[0]
    print(template.shape[1])
    if template.shape[1] == 47: #7.2mm x 5.4mm
        xscale = ( (original_image_resolutionx / original_image_width) * (7.05/47))
        yscale = ( (original_image_resolutiony / original_image_height) * (5.25/35))
        print("7.2mm x 5.4mm detected ┬─┬ノ( º _ ºノ)")

    elif template.shape[1] == 35:
        xscale = ( (original_image_resolutionx / original_image_width) * (5.25/35))
        yscale = ( (original_image_resolutiony / original_image_height) * (5.25/35))
        print("5.4mm x 5.4mm detected ┬─┬ノ( º _ ºノ)")
    
    elif template.shape[1] == 23:
        xscale = ( (original_image_resolutionx / original_image_width) * (3.45/23))
        yscale = ( (original_image_resolutiony / original_image_height) * (3.45/23))
        print("3.84mm x 3.84mm detected")

    elif template.shape[1] == 15:
        xscale = ( (original_image_resolutionx / original_image_width) * (2.25/15))
        yscale = ( (original_image_resolutiony / original_image_height) * (2.25/15))
        print("2.56mm x 2.56mm detected")

    else:
        print("Size not recognized... Skipping (╯°□°)╯︵ ┻━┻")
    return xscale,yscale


def loadSnippit(original_image, original_image_width, original_image_height): #loads templates
    template_folder = input("Enter snippit folder. Remember that the patterned and unpatterned counter parts must share the same name in separate folders:")
    # Paths to the template images
    template_paths = glob.glob(str(template_folder) + "/*.png") 




    # Load the templates
    templates = []
    templateNames = []

    #make into case
    for template_path in template_paths: #photo size detections
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Template image {template_path} could not be loaded (╯°□°)╯︵ ┻━┻")
        
        
        else:
            xscale,yscale = calcScale(original_image,template,original_image_width,original_image_height)
            
            

            #properly scales template to  original
            template = cv2.resize(template, (round(template.shape[1] * yscale), round(template.shape[0] * xscale)), interpolation=cv2.INTER_CUBIC)
            templates.append(template)
            templateNames.append(os.path.basename(template_path))

    return templates, templateNames, xscale, yscale, template_folder


#STUFF FOR PHASE DELAY

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, A):
    return A * np.exp(-(((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2))))

def fit_gaussian_to_power(power_array, initial_guess = None):
    # Extract millimeter coordinates from the power array
    x_coords_mm = power_array[0, 1:]  # x coordinates in mm
    y_coords_mm = power_array[1:, 0]  # y coordinates in mm

    # Create grid of coordinates
    X_mm, Y_mm = np.meshgrid(x_coords_mm, y_coords_mm)
    
    # Flatten the arrays for curve fitting
    xdata = np.vstack((X_mm.ravel(), Y_mm.ravel()))
    ydata = power_array[1:, 1:].ravel()

    if initial_guess is None:
        print("cheesin")
        initial_guess = [0, 0, 2.8, 2.8, power_array[1:, 1:].max()]

    params = np.array(initial_guess)
    max_iterations = 10
    tol = 1e-8
    for i in range(max_iterations):
        # Fit Gaussian model
        new_params, _ = curve_fit(lambda x, x0, y0, sigma_x, sigma_y, A: gaussian_2d(x[0], x[1], x0, y0, sigma_x, sigma_y, A),
                                 xdata, ydata, p0=params)
        
        # Check for convergence
        param_change = np.linalg.norm(new_params - params)
        if param_change < tol:
            print(f"Converged after {i + 1} iterations.")
            break
        
        # Update parameters for the next iteration
        params = new_params

    x0, y0, sigma_x, sigma_y, A = params
    fitted_gaussian = gaussian_2d(X_mm, Y_mm, x0, y0, sigma_x, sigma_y, A)
    
    
    
    # Compute residuals
    residuals = power_array[1:, 1:] - fitted_gaussian
    percent_diff = (residuals / power_array[1:, 1:]) * 100
    
    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot original data
    c = axs[0].pcolormesh(X_mm, Y_mm, power_array[1:, 1:], shading='auto', cmap='viridis')
    axs[0].set_title('Original Data')
    fig.colorbar(c, ax=axs[0])
    
    # Plot fitted Gaussian
    c = axs[1].pcolormesh(X_mm, Y_mm, fitted_gaussian, shading='auto', cmap='viridis')
    axs[1].set_title('Fitted Gaussian')
    fig.colorbar(c, ax=axs[1])
    
    # Plot residuals
    c = axs[2].pcolormesh(X_mm, Y_mm, percent_diff, shading='auto', cmap='coolwarm', vmin=-100, vmax=100)
    axs[2].set_title('Percent Difference')
    cbar = fig.colorbar(c, ax=axs[2])
    cbar.set_label('Percent Difference')
    
    plt.tight_layout()
    plt.show()


    print(x0)
    print(y0)

    return x0, y0, sigma_x, sigma_y, A, params

def plot_beam_analysis(folder, index):
    power_array = getPowerCSV(folder, index)
    x0_mm, y0_mm, sigma_x, sigma_y, A, params = fit_gaussian_to_power(power_array)

    x_min, x_max = x0_mm - 3 * sigma_x, x0_mm + 3 * sigma_x
    y_min, y_max = y0_mm - 3 * sigma_y, y0_mm + 3 * sigma_y
    
    return x0_mm, y0_mm, x_max, x_min,y_max,y_min


def mostCenter(orignal,templateCenters): #finds template that is closest to center
    counter = -1 #keeps track of its position in array
    num = 0
    py,px = orignal.shape
    originalCenter = (px/2,py/2)

    min_distance = float('inf')
    closest_point = None
    
    for p in templateCenters:
        distance = math.sqrt((p[0] - originalCenter[0]) ** 2 + (p[1] - originalCenter[1]) ** 2)
        counter += 1
        if distance < min_distance:
            min_distance = distance
            closest_point = p
            num = counter
    return num


def mmFromCenter(px,py, original, sides, templateCenters): #finds shift of snippits
   
    height, width = original.shape

    # Calculate center pixel coordinates
    center_x = px
    center_y = py

    # Display the center pixel coordinates
    print(f"Center Pixel Coordinates: ({center_x}, {center_y})")
    mmPerPixel = sides/width
    print(f"mm per pixel: {mmPerPixel}mm")
    mmXShift = []
    mmYShift = []

    for center in templateCenters: #dont need the if, just left for if you want to see where photo is
        if(center[0]>= center_x):
            xshift = center[0] - center_x
        if(center[0]<center_x):
            xshift = center[0] - center_x
        if(center[1]>=center_y): #these must be flipped bc larger pixel number, is lower in photo
            yshift = center_y - center[1]
        if(center[1]<center_y):
            yshift = center_y - center[1]
            #pixels to mm
        mmXShift.append(xshift * mmPerPixel)
        mmYShift.append(yshift * mmPerPixel)
    return mmXShift, mmYShift

def make3d(datasets,labels,outputDir, minX = None, maxX = None, minY = None, maxY = None):
    all_x_values = []
    all_y_values = []
    all_z_values = []


    for x_values, y_values, plot_data in datasets:
        X, Y = np.meshgrid(x_values, y_values)
        Z = plot_data.flatten()
        if minX is not None and maxX is not None:
            mask_x = (X.flatten() >= minX) & (X.flatten() <= maxX)
        else:
            mask_x = np.ones_like(X.flatten(), dtype=bool)

        if minY is not None and maxY is not None:
            mask_y = (Y.flatten() >= minY) & (Y.flatten() <= maxY)
        else:
            mask_y = np.ones_like(Y.flatten(), dtype=bool)
        
        mask = mask_x & mask_y

        valid_X = X.flatten()[mask]
        valid_Y = Y.flatten()[mask]
        valid_Z = Z[mask]
        valid_Z = valid_Z[~np.isnan(valid_Z)]  # Remove NaN values

        all_z_values.extend(valid_Z)
        all_x_values.extend(valid_X)
        all_y_values.extend(valid_Y)

    # Calculate the min and max Z values for normalization
    vmin = min(all_z_values)
    vmax = max(all_z_values)
    # Create 3D scatter plot
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')

    for (x_values, y_values, plot_data), label in zip(datasets, labels):
        X, Y = np.meshgrid(x_values, y_values)
        Z = plot_data
        if minX is not None and maxX is not None:
                mask_x = (X.flatten() >= minX) & (X.flatten() <= maxX)
        else:
            mask_x = np.ones_like(X.flatten(), dtype=bool)

        if minY is not None and maxY is not None:
            mask_y = (Y.flatten() >= minY) & (Y.flatten() <= maxY)
        else:
            mask_y = np.ones_like(Y.flatten(), dtype=bool)
        
        mask = mask_x & mask_y

        valid_X = X.flatten()[mask]
        valid_Y = Y.flatten()[mask]
        valid_Z = Z.flatten()[mask]
        #norm = SymLogNorm(linthresh=1, linscale=1, vmin=vmin, vmax=vmax)
        sc3d = ax3d.scatter(valid_X, valid_Y, valid_Z, c=valid_Z, cmap='viridis', s=.5, label=label)
        sc3d.set_norm(plt.Normalize(vmin=vmin, vmax=vmax))

    cbar3d = fig3d.colorbar(sc3d, ax=ax3d, label='Delay')
    ax3d.set_xlabel('X Coordinates')
    ax3d.set_ylabel('Y Coordinates')
    ax3d.set_zlabel('Z Values')
    ax3d.set_title('Combined 3D Scatter Plot')

    ax3d.set_xlim(min(all_x_values) - 1, max(all_x_values) + 1)
    ax3d.set_ylim(min(all_y_values) - 1, max(all_y_values) + 1)
    ax3d.set_zlim(vmin - 1, vmax + 1)

    imgOutput3d = os.path.join(outputDir, "combined_3d_plot.png")
    fig3d.savefig(imgOutput3d, dpi=800)
    plt.show()

    try:
        minX = float(input("Enter new minX (or press Enter to skip): ") or minX)
        maxX = float(input("Enter new maxX (or press Enter to skip): ") or maxX)
        minY = float(input("Enter new minY (or press Enter to skip): ") or minY)
        maxY = float(input("Enter new maxY (or press Enter to skip): ") or maxY)
        # Replot the graph with new values
        make3d(datasets,labels,outputDir, minX, maxX, minY, maxY)
    except ValueError:
        print("Invalid input. Skipping replotting.")

def makeScatterPlot(datasets, labels, outputDir):
    all_x_values = []
    all_y_values = []
    all_z_values = []


    for x_values, y_values, plot_data in datasets:
        X, Y = np.meshgrid(x_values, y_values)
        Z = plot_data.flatten()
        valid_Z = Z[~np.isnan(Z)]  # Remove NaN values
        all_z_values.extend(valid_Z)
        all_x_values.extend(X.flatten())
        all_y_values.extend(Y.flatten())

    # Calculate the min and max Z values for normalization
    vmin = min(all_z_values)
    vmax = max(all_z_values)

    fig2d, ax2d = plt.subplots()

    for (x_values, y_values, plot_data), label in zip(datasets, labels):
        X, Y = np.meshgrid(x_values, y_values)
        Z = plot_data

        norm = SymLogNorm(linthresh=1, linscale=1, vmin=vmin, vmax=vmax)
        sc2d =plt.scatter(X.flatten(), Y.flatten(), c=Z.flatten(), cmap='viridis', s=.05, label=label, norm=norm)
        sc2d.set_norm(plt.Normalize(vmin=vmin, vmax=vmax))


    ax2d.set_facecolor('black')
    cbar2d = fig2d.colorbar(sc2d, ax=ax2d, label='Delay')
    ax2d.set_xlabel('X Coordinates')
    ax2d.set_ylabel('Y Coordinates')
    ax2d.set_title('Combined 2D Scatter Plot')

    ax2d.set_xlim(min(all_x_values) - 1, max(all_x_values) + 1)
    ax2d.set_ylim(min(all_y_values) - 1, max(all_y_values) + 1)

    ax2d.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
    ax2d.yaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
    cbar2d.set_ticks(np.linspace(vmin, vmax, 10))

    imgOutput2d = os.path.join(outputDir, "combined_2d_plot.png")
    fig2d.savefig(imgOutput2d, dpi=800)



def getPowerArray(folder, outputDir, xshift, yshift):
    csvFiles = glob.glob(os.path.join(folder, "*.csv"))
    datasets = []
    labels = []

    if not csvFiles:
        print("Oops, no files found.")
        return

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for counter, file in enumerate(csvFiles):
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        labels.append(name)

        with open(file, 'r') as csvfile:
            lines = csvfile.readlines()

        # Find the starting line after "Spots Y"
        start_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Spots Y"):
                start_line = i + 1  # Start processing from the next line
                break

        if start_line is None:
            print(f"No 'Spots Y' line found in {file}. Skipping.")
            continue
        
        # Read data from the identified start line
        csv_reader = csv.reader(lines[start_line:])
        data = list(csv_reader)

        # Process the data to extract coordinates
        x_values = [float(val.strip()) for val in data[0][1:] if val.strip()]  # All elements except the first from the first row
        y_values = [float(row[0].strip()) for row in data[1:] if row[0].strip()]  # All elements except the first from the first column
        plot_data = np.array([[float(val.strip()) if val.strip() else np.nan for val in row[1:]] for row in data[1:]])

        plot_data = plot_data[:, :-1]
        # Apply shifts
        y_values = [-y for y in y_values] #camera is upside down, must flip
        x_values = [x + xshift[counter] for x in x_values]
        y_values = [y + yshift[counter] for y in y_values]

        # Adjust x_values and y_values to match the plot_data's dimensions
        x_values = x_values[:plot_data.shape[1]]
        y_values = y_values[:plot_data.shape[0]]

        datasets.append((x_values, y_values, plot_data))

    # Create the scatter plot
    makeScatterPlot(datasets, labels, outputDir)
    make3d(datasets, labels, outputDir )
    print(f".csv files and images saved to {outputDir}")



def phase(xshift, yshift):
    folderPath = input("Enter the directory containing .csv files:")
    outputPath = input("Enter the output directory for images: ")
    getPowerArray(folderPath, outputPath, xshift, yshift)
#END OF STUFF FOR PHASE DELAY


def getPowerCSV(folder, num):
    # Get a list of all CSV files in the folder
    csvFiles = glob.glob(os.path.join(folder, "*.csv"))
    
    # Select the file at the specified index
    file = csvFiles[num]
    
    with open(file, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    print("Opened and read file " + file)
    first_row_values = [
        -3.450, -3.300, -3.150, -3.000, -2.850, -2.700, -2.550, -2.400, -2.250, -2.100, 
        -1.950, -1.800, -1.650, -1.500, -1.350, -1.200, -1.050, -0.900, -0.750, -0.600, 
        -0.450, -0.300, -0.150, 0.000, 0.150, 0.300, 0.450, 0.600, 0.750, 0.900, 
        1.050, 1.200, 1.350, 1.500, 1.650, 1.800, 1.950, 2.100, 2.250, 2.400, 
        2.550, 2.700, 2.850, 3.000, 3.150, 3.300, 3.450
    ]
    #negated to flip into right axis. Camera is upside down
    first_col_values = [2.550, 2.4, 2.25, 2.1, 1.950, 1.80, 1.650, 1.5, 1.35, 1.2, 1.50, 0.9, 0.75,
                    0.60, 0.45, 0.3, 0.15, 0.0, -0.15, -0.3, -0.45, -0.6, -0.75, -0.9, -1.050,
                     -1.2, -1.350, -1.5, -1.65, -1.8, -1.95, -2.1, -2.25, -2.4, -2.55]
    
    # Create a 48x36 array with the first row set to the specified values
    array = np.zeros((36, 48))
    array[0, 1:] = first_row_values
    array[1:36, 0] = first_col_values

    for i, line in enumerate(lines):
        for j, value in enumerate(line):
            array[i + 1, j + 1] = float(value)
    
    return array




def powerMosiac(locations, original_image, original_image_width, original_image_height,output):
    newtemplates, _, __, _, folder = loadSnippit(original_image, original_image_width, original_image_height)
    data = zip(newtemplates, locations)
    
    # Initialize the canvas and the alpha_accumulation canvas
    canvas = np.zeros_like(original_image, dtype=np.float32)
    alpha_accum = np.zeros_like(original_image, dtype=np.float32)
    
    num_snippets = len(newtemplates)
    alpha_step = 1.0 / num_snippets
    
    for i, (img, corner) in enumerate(data):
        x, y = corner
        h, w = img.shape
        
        alpha = 1.0 - (i * alpha_step)  # Decrease alpha for each successive snippet
        
        # Convert img to float32 for proper blending
        img = img.astype(np.float32)
        
        # Blend the snippet with the canvas and update alpha_accum
        canvas[y:y+h, x:x+w] += alpha * img
        alpha_accum[y:y+h, x:x+w] += alpha
    
    # Avoid division by zero
    alpha_accum[alpha_accum == 0] = 1
    
    # Normalize the canvas with the accumulated alpha values
    final_canvas = canvas / alpha_accum
    
    # Clip values to the valid range and convert back to original image dtype
    final_canvas = np.clip(final_canvas, 0, 255).astype(original_image.dtype)
    save = os.path.join(output,"powerMosiac.jpg")
    cv2.imwrite(save, final_canvas)
    return final_canvas, folder


#centroid stuff
def loadPixels(original_image):
    template_folder = input("Enter centroid folder. They need the same name as their patterned counterpart:")
    template_paths = glob.glob(str(template_folder) + "/*.png") 
    original_image_width = 17 #mm
    original_image_resolutionx = original_image.shape[1]
    original_image_resolutiony = original_image.shape[0]

    templates = []
    templateNames = []

    for template_path in template_paths: #photo size detections
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Template image {template_path} could not be loaded.")
            
        if template.shape[1] == 1440:
            xscale = ( 1.0 / ((original_image_resolutionx / original_image_width) * (7.2/1440.0)))
            yscale = ( 1.0 / ((original_image_resolutiony / original_image_width) * (5.4/1080.0)))
            templates.append(template)

    return templates, templateNames, xscale, yscale

def centroid(locations, original_image):
    new_templates, _, xscale, yscale = loadPixels(original_image)
    scaled_locations = [(round( x * xscale), round(y * yscale)) for x, y in locations]
    data = zip(new_templates, scaled_locations)
    canvas = np.zeros((round(original_image.shape[0] * xscale), round(original_image.shape[1] * yscale)), dtype=original_image.dtype)
    for img, corner in data:
        x,y = corner
        h,w = img.shape
        canvas[y:y+h, x:x+w] = img

    cv2.imwrite('centroidMosaic.jpg', canvas)



def ImageMatch():
    original_image_path = input("Enter pattern path")
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    patternWidth = int(input("Enter width of pattern in mm"))
    patternLength =int(input("Enter length of pattern in mm "))


    output_folder = input("Enter the output folder: ")
    output_locations = os.path.join(output_folder, "locationslist.txt")


    if original_image is None:
        raise ValueError(f"No image found at {original_image_path}")
        exit()

    templates, templateNames, xscale, yscale,_ = loadSnippit(original_image,patternWidth, patternLength)







    locations = []
    loacationStrings = []
    centers = []

    #can do x plus half, y plus half. that gives center.
    for i, template in enumerate(templates):
        location = find_template_location(original_image, template)
        locations.append(location)
        centers.append((location[0] + template.shape[1]/2, location[1] + template.shape[0]/2 ))
        locationString = f"template {templateNames[i]} found with center {centers[i]} "

        print(locationString)
        loacationStrings.append(locationString)

    with open(output_locations, 'w', newline='') as file:
        for i, string in enumerate(loacationStrings):
            file.write(f"{string}\n") 

    # Visualize the locations by drawing rectangles on the original image
    output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    # Define a list of colors (as BGR tuples for OpenCV)
    colors = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (0,165,255),     #Orange
    (128,0,128)]     #purple

    for i, template in enumerate(templates):
        top_left = locations[i]
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        
        # Select color by cycling through the colors list
        color = colors[i % len(colors)]
        
        # Draw rectangle with the selected color
        cv2.rectangle(output_image, top_left, bottom_right, color, 2)

    # Save the output image with rectangles
    output_path = "./output/marked.jpg"
    cv2.imwrite(output_path, output_image)

    print(f"Output image with marked locations saved at {output_path}")

    mosaic = None

    answer = input("Create Mosaic :D (Y/N)? ")

    if answer == 'Y':
        mosaic, folder = powerMosiac(locations, original_image, patternWidth, patternLength,output_folder)
    
    #answer = input("Centroid locations :D (Y/N)?")

    #if answer == 'Y':
        #centroid(locations, original_image)
    
    answer = input("Phase Delay Scatter (Y/N)?")
    if answer == 'Y':
        x0 = 0
        y0 = 0
        answer = input ("Full mosiac?(Y/N) ONLY LARGEST")
        if answer == 'Y':
            if mosaic is not None: #if you made a mosiaci
                
                index = mostCenter(original_image, centers)
                x0, y0, max_x,min_x,max_y,min_y =plot_beam_analysis(folder,index)
                #find the power array of index
            else:
                print("A mosaic was not created!")

        h, w = original_image.shape
        py = h/2
        px = w/2
        xshift, yshift = mmFromCenter(px, py,original_image, patternWidth, centers)
        xshift = [x + x0 for x in xshift]
        yshift = [y+y0 for y in yshift]  
        phase(xshift, yshift)
        
    



ImageMatch()







    

