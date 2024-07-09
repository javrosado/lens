import csv
import glob
import cv2
import numpy as np
from PIL import Image
import os

def getPowerArray(folder, outputDir):

    csvFolder = glob.glob(str(folder) + "/*/") 
    counter = 1
    if not os.path.exists(outputDir): #if dir already exists, dont make a new one
        makedir(outputDir)

    for folder in csvFolder:
        csvFiles = glob.glob(str(folder) + "/*.csv") 
        if not csvFiles: 
            print("Oops, no files :()")
            exit()

        darkFile = csvFiles[0]    
        with open(darkFile , 'r') as file:
            lines = file.readlines()
        print("Opened and read file " + file.name)

        csv_Line_Start = None
        for i, line in enumerate(lines): #for every line, check for power points y, next line is array
            if line.strip().startswith("Power points Y"):
                csv_Line_Start = i +1 
                #print("Start Line found!")
                break

        if csv_Line_Start == None:
            print("No Power array found D:")
            exit()

        csv_Lines = lines[csv_Line_Start:]

        darkData= []

        for line in csv_Lines: #for every line, take values
            #line split thing?

            row = [int(value.strip()) for value in line.split(",") if value.strip()]
            darkData.append(row)


        lightFile = csvFiles[1]    
        with open(lightFile , 'r') as file:
            lines = file.readlines()
        print("Opened and read file " + file.name)

        csv_Line_Start = None
        for i, line in enumerate(lines): #for every line, check for power points y, next line is array
            if line.strip().startswith("Power points Y"):
                csv_Line_Start = i +1 
                #print("Start Line found!")
                break

        if csv_Line_Start == None:
            print("No Power array found D:")
            exit()

        csv_Lines = lines[csv_Line_Start:]

        lightData= []

        for line in csv_Lines: #for every line, take values
            #line split thing?

            row = [int(value.strip()) for value in line.split(",") if value.strip()]
            lightData.append(row)

        lightArray = np.array(lightData)
        darkArray = np.array(darkData)
        finalData = lightArray - darkArray
        finalData = np.abs(finalData) #maybe make it 0

        

    #make it so that is is name, outputfile HEERE
        output_file_csv =os.path.join(outputDir,'output' + str(counter) + ".csv")

        with open(output_file_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(finalData)

        makeImage(finalData, counter, outputDir)
        counter +=1

        print(".csv and images saved to " + str(outputDir))

def makeImage(data, counter, outputDir):
    normed = np.array(data) #normalizes the photo, som that numbers above 255, can be dislayed
    normed = (normed - np.min(normed)) * (255.0/(np.max(normed) - np.min(normed)))

    
    answer= input("Is this a patterned photo? (Y/N)")
    if answer == 'Y':
        
        non_black = normed[normed>0] 
        normed[(normed>0) ] =255
        normed = normed.astype(np.uint8) #makes photo
        img = Image.fromarray(normed, mode='L')  # 'L' mode for grayscale

        

        
    # Save or display the image
        imgOutput = os.path.join(outputDir, 'output' + str(counter) + ".png" )



    else: 
        normed = normed.astype(np.uint8) #makes photo
        img = Image.fromarray(normed, mode='L')  # 'L' mode for grayscale
    # Save or display the image
        imgOutput = os.path.join(outputDir, 'output' + str(counter) + ".png" )



    img.save(imgOutput)  # Save the image
    img.show() 

def makedir(path):
    os.mkdir(path)








def main():
    folderPath = input("Enter the .csv directory")
    outputPath = input("Enter the output directory")
    getPowerArray(folderPath, outputPath)


main()
