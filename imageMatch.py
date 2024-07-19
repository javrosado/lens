import math
import cv2
import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def find_template_location(original, template): #finds template
    result = cv2.matchTemplate(original, template, cv2.TM_CCOEFF_NORMED)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc



def calcScale(original,template)


def loadSnippit(original_image, original_image_width, original_image_height): #loads templates
    template_folder = input("Enter snippit folder. Remember that the patterned and unpatterned counter parts must share the same name in separate folders:")
    # Paths to the template images
    template_paths = glob.glob(str(template_folder) + "/*.png") 


    #pattern dimensions
    original_image_resolutionx = original_image.shape[1]
    original_image_resolutiony = original_image.shape[0]




    # Load the templates
    templates = []
    templateNames = []

    #make into case
    for template_path in template_paths: #photo size detections
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Template image {template_path} could not be loaded (╯°□°)╯︵ ┻━┻")
    #WIERD, SO IS IT .153 * 47 lenses? That is 7.2 answays
        elif template.shape[1] == 47: #7.2mm x 5.4mm
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
            continue
        
       

        #properly scales template to  original
        template = cv2.resize(template, (round(template.shape[1] * yscale), round(template.shape[0] * xscale)), interpolation=cv2.INTER_CUBIC)
        templates.append(template)
        templateNames.append(os.path.basename(template_path))

    return templates, templateNames, xscale, yscale


#STUFF FOR PHASE DELAY

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, A):
    return A * np.exp(-(((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2))))

def fit_gaussian_to_power(power_array):
    x = np.arange(power_array.shape[1])
    y = np.arange(power_array.shape[0])
    X, Y = np.meshgrid(x, y)
    
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = power_array.ravel()
    
    initial_guess = [power_array.shape[1] / 2, power_array.shape[0] / 2, 1, 1, power_array.max()]
    
    params, _ = curve_fit(lambda x, x0, y0, sigma_x, sigma_y, A: gaussian_2d(x[0], x[1], x0, y0, sigma_x, sigma_y, A),
                          xdata, ydata, p0=initial_guess)
    
    x0, y0, sigma_x, sigma_y, A = params
    return x0, y0

def findCenterBeam(powerCSVS):
    centers = []

    for csv in powerCSVS:
        power_df = pd.read_csv(power_file, header=None)
        power_array = power_df.values

        x0, y0 = fit_gaussian_to_power(power_array)
    
        # Store the center for comparison
        centers.append((x0, y0, power_file))


def mostCenter(orignal,templateCenters):
    counter = 1
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
    print(num)
    return min_distance, closest_point[0], closest_point[1]


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
        if(center[1]>=center_y):
            yshift = center_y - center[1]
        if(center[1]<center_y):
            yshift = center_y - center[1]
            #pixels to mm
        mmXShift.append(xshift * mmPerPixel)
        mmYShift.append(yshift * mmPerPixel)
    return mmXShift, mmYShift


def makeScatterPlot(datasets, labels, outputDir):
    all_x_values = []
    all_y_values = []
    all_z_values = []

    for (x_values, y_values, plot_data), label in zip(datasets, labels):
        X, Y = np.meshgrid(x_values, y_values)
        Z = plot_data

        plt.scatter(X.flatten(), Y.flatten(), c=Z.flatten(), cmap='viridis', s=.05, label=label)

        all_x_values.extend(x_values)
        all_y_values.extend(y_values)
        flattened_Z = Z.flatten()
        valid_Z = flattened_Z[~np.isnan(flattened_Z)]  # Remove NaN values

        all_z_values.extend(valid_Z)
    # Normalize the color scale
    vmin = min(all_z_values)  
    vmax = max(all_z_values) 

    for i in range(len(datasets)):
        plt.gca().collections[i].set_norm(plt.Normalize(vmin=vmin, vmax=vmax))

    plt.gca().set_facecolor('black')  
    plt.colorbar(label='Delay')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Combined Scatter Plot')
   

    plt.xlim(min(all_x_values) -1 , max(all_x_values)+1)
    plt.ylim(min(all_y_values) - 1, max(all_y_values) +1)



    imgOutput = os.path.join(outputDir, "combined_plot.png")
    plt.savefig(imgOutput, dpi = 700)
    print("after")


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
        print(len(xshift))
        x_values = [x + xshift[counter] for x in x_values]
        y_values = [y + yshift[counter] for y in y_values]

        # Adjust x_values and y_values to match the plot_data's dimensions
        x_values = x_values[:plot_data.shape[1]]
        y_values = y_values[:plot_data.shape[0]]

        datasets.append((x_values, y_values, plot_data))

    # Create the scatter plot
    makeScatterPlot(datasets, labels, outputDir)
    print(f".csv files and images saved to {outputDir}")

def phase(xshift, yshift):
    folderPath = input("Enter the directory containing .csv files:")
    outputPath = input("Enter the output directory for images: ")
    getPowerArray(folderPath, outputPath, xshift, yshift)
#END OF STUFF FOR PHASE DELAY

def powerMosiac(locations, original_image, original_image_width, original_image_height,output):
    newtemplates, _, __, _ = loadSnippit(original_image, original_image_width, original_image_height)
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
    return final_canvas


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

    templates, templateNames, xscale, yscale = loadSnippit(original_image,patternWidth, patternLength)







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
    for i, template in enumerate(templates):
        top_left = locations[i]
        
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        cv2.rectangle(output_image, top_left, bottom_right, (0, 255, 0), 2)

    # Save the output image with rectangles
    output_path = "./output/marked.jpg"
    cv2.imwrite(output_path, output_image)

    print(f"Output image with marked locations saved at {output_path}")

    mosaic = None

    answer = input("Create Mosaic :D (Y/N)? ")

    if answer == 'Y':
        mosaic = powerMosiac(locations, original_image, patternWidth, patternLength,output_folder)
    
    #answer = input("Centroid locations :D (Y/N)?")

    #if answer == 'Y':
        #centroid(locations, original_image)
    
    answer = input("Phase Delay Scatter (Y/N)?")
    if answer == 'Y':

        answer = input ("Full mosiac?(Y/N)")
        if answer == 'Y':
            if mosaic is not None:
                _,px,py = mostCenter(original_image, centers)
            else:
                print("A mosaic was not created!")
        else:
            h, w = original_image.shape
            py = h/2
            px = w/2
        
        xshift, yshift = mmFromCenter(px, py,original_image, patternWidth, centers)

        phase(xshift, yshift)
    



ImageMatch()







    

