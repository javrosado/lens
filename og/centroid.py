import csv
import glob
import cv2
import numpy as np
from PIL import Image
import os

def getPowerArray(folder, outputDir):

    csvFiles = glob.glob(str(folder) + "/*.csv") 
    counter = 1

    if not csvFiles: 
        print("Oops, no files :()")
        exit()

    if not os.path.exists(outputDir): #if dir already exists, dont make a new one
        makedir(outputDir)

    for file in csvFiles:#for  every file,in the csvfolder, read them
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        

        
        with open(file , 'r') as file:
            lines = file.readlines()
        print("Opened and read file " + file.name)

        csv_Line_Start = None
        for i, line in enumerate(lines): #for every line, check for power points y, next line is array
            if line.strip().startswith("Centroids Y"):
                csv_Line_Start = i +1 
                #print("Start Line found!")
                break

        if csv_Line_Start == None:
            print("No Power array found D:")
            exit()
    
        csv_Lines = lines[csv_Line_Start:]

        data= []

        for line in csv_Lines: #for every line, take values
            #line split thing?

           for line in csv_Lines: #for every line, take values FIX THIS FOR FLOAT
            row = [float(value.strip().replace('NaN', '0') if value.strip() else '0') for value in line.split(",")]
            row.pop()
            row = [int(round(value)) for value in row]  # Convert float values to integers
            pairs = [(row[i], row[i + 1]) for i in range(0, len(row), 2)]
            data.append(pairs)
    
    
       

#make it so that is is name, outputfile HEERE
        output_file_csv =os.path.join(outputDir,'output' + str(counter) + ".csv")

        with open(output_file_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

            # Get the base name of the file (e.g., 'imageMatch.py')

# Split the base name into name and extension (e.g., ('imageMatch', '.py'))

        makeImage(data, name, outputDir)
        counter +=1

    print(".csv and images saved to " + str(outputDir))

def makeImage(data, name, outputDir):
    # Determine the size of the image based on the maximum coordinates in data
    max_x = 1440
    max_y = 1080

    # Create a blank (black) image
    img = np.zeros((max_y , max_x), dtype=np.uint8)

    # Set specified pixels to white
    for row in data:
        for coord in row:
            x, y = coord
            img[y, x] = 255  # Set the pixel at (x, y) to white

    # Convert the NumPy array to an image and save or display it
    img = Image.fromarray(img, mode='L')  # 'L' mode for grayscale
    imgOutput = os.path.join(outputDir, name + ".png")

    img.save(imgOutput)  # Save the image
    img.show() 

def makedir(path):
    os.mkdir(path)







def main():
    folderPath = input("Enter the .csv directory")
    outputPath = input("Enter the output directory")
    getPowerArray(folderPath, outputPath)


main()