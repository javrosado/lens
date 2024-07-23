import csv
import glob
import os
import numpy as np
from PIL import Image

def getPowerArray(folder, outputDir):
    csvFiles = glob.glob(os.path.join(folder, "*.csv"))
    counter = 1

    if not csvFiles:
        print("Oops, no files :()")
        exit()

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    total_data = []

    # First pass: Read CSVs and add all data to total_data
    answer = input("Make it white? (Y/N)")
    for file in csvFiles:
        with open(file, 'r') as f:
            lines = f.readlines()


        csv_Line_Start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Power points Y"):
                csv_Line_Start = i + 1
                break

        if csv_Line_Start is None:
            print(f"No Power array found in {file}")
            continue

        csv_Lines = lines[csv_Line_Start:]

        for line in csv_Lines:
            try:
                row = [int(value.strip().replace('NaN', '0') if value.strip() else '0') for value in line.split(",")]
                row.pop()  # Remove last element
            except ValueError as e:
                row = [int(float((value.strip().replace('nan', '0') if value.strip() else '0'))) for value in line.split(",")]
            total_data.extend(row)

    # Convert total_data to numpy array and find global min and max
    total_data = np.array(total_data)
    global_min = 0
    global_max = np.max(total_data)

    # Second pass: Normalize data and create images
    for file in csvFiles:
        with open(file, 'r') as f:
            lines = f.readlines()

        print("Opened and read file " + file)

        csv_Line_Start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Power points Y"):
                csv_Line_Start = i + 1
                break

        if csv_Line_Start is None:
            print(f"No Power array found in {file}")
            continue

        csv_Lines = lines[csv_Line_Start:]

        data = []
        for line in csv_Lines:
            try:
                row = [int(value.strip().replace('NaN', '0') if value.strip() else '0') for value in line.split(",")]
                row.pop()  # Remove last element
            except ValueError as e:
                row = [int(float((value.strip().replace('nan', '0') if value.strip() else '0'))) for value in line.split(",")]
            data.append(row)

        # Normalize data
        data = np.array(data)
        normed = (data - global_min) * (255.0 / (global_max - global_min))
        name = os.path.splitext(os.path.basename(file))[0]

        output_file_csv = os.path.join(outputDir, f'{name}.csv')
        with open(output_file_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        # Create and save image
        makeImage(normed, name, outputDir, answer)
        counter += 1

    print(".csv and images saved to " + str(outputDir))

def makeImage(normed, name, outputDir, answer):
    if answer.upper() == 'Y':
        normed[normed > 0] = 255
    normed = normed.astype(np.int8)
    img = Image.fromarray(normed, mode='L')
    imgOutput = os.path.join(outputDir, name + ".png")
    img.save(imgOutput)

def main():
    folderPath = input("Enter the .csv directory: ")
    outputPath = input("Enter the output directory: ")
    getPowerArray(folderPath, outputPath)

main()