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
        with open(file , 'r') as file:
            lines = file.readlines()
        print("Opened and read file " + file.name)

       
        csv_Lines = lines[0:]

        data= []

        for line in csv_Lines: #for every line, take values
            #line split thing?

            row = [float(value.strip()) for value in line.split(",") if value.strip()]
            data.append(row)
    
       

#make it so that is is name, outputfile HEERE
        output_file_csv =os.path.join(outputDir,'output' + str(counter) + ".csv")

        with open(output_file_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        makeImage(data, counter, outputDir)
        counter +=1

    print(".csv and images saved to " + str(outputDir))

def makeImage(data, counter, outputDir):
    normed = np.array(data) #normalizes the photo, som that numbers above 255, can be dislayed
    normed = (normed - np.min(normed)) * (255.0/(np.max(normed) - np.min(normed)))

    
    answer= input("Is this a patterned photo? (Y/N)")
    if answer == 'Y':
        #out of the non black pixels, find the bottom 20percent, make them black
        non_black = normed[normed>0] 
        bottom_percentile = np.percentile(non_black,20)
        normed[(normed>0) & (normed<= bottom_percentile)] =0
#update which are black again, out of the non black ones left, the top 60% is made white
        non_black = normed[normed>0]
        top_percentile = np.percentile(non_black, 50)
        normed[(normed>0) & (normed>= top_percentile)] =255



        normed = normed.astype(np.uint8) #makes photo
        img = Image.fromarray(normed, mode='L')  # 'L' mode for grayscale

        

        
    # Save or display the image
        imgOutput = os.path.join(outputDir, 'output' + str(counter) + ".png" )
        img = img.resize((img.width *1500 , img.height * 1500), Image.Resampling.LANCZOS) #upscales

        img = img.resize((int(img.width / 300 ) , int(img.height / 300 )), Image.Resampling.NEAREST)

        

    
        pixels = img.load()  # Create the pixel map
        for i in range(img.size[0]):  # For every pixel:
            for j in range(img.size[1]):
                c = pixels[i, j]
                if c > 20:  # If the pixel is not black
                    pixels[i, j] = 255  # Change it to white


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
