import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def getPowerArray(folder, outputDir):
    csvFiles = glob.glob(os.path.join(folder, "*.csv"))
    counter = 1

    if not csvFiles:
        print("Oops, no files found.")
        return

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for file in csvFiles:
        print("RUns")
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)

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
        plot_data = np.array([[float(val.strip()) if val.strip() else 0 for val in row[1:]] for row in data[1:]])

        # Remove the last row and last column from plot_data
        plot_data = plot_data[:-1, :-1]

        # Adjust x_values and y_values to match the plot_data's dimensions
        x_values = x_values[:plot_data.shape[1]]
        y_values = y_values[:plot_data.shape[0]]

        # Create output CSV file
        output_file_csv = os.path.join(outputDir, f'output{counter}.csv')
        np.savetxt(output_file_csv, plot_data, delimiter=',')

        # Create the scatter plot
        makeScatterPlot(x_values, y_values, plot_data, name, outputDir)

        counter += 1

    print(f".csv files and images saved to {outputDir}")

def makeScatterPlot(x_values, y_values, data, name, outputDir):
    plt.clf()
    plt.cla()
    X, Y = np.meshgrid(x_values, y_values)
    Z = data

    plt.scatter(X.flatten(), Y.flatten(), c=Z, cmap='viridis', vmin=-45, vmax=45)
    plt.colorbar(label='Intensity')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title(f'Scatter Plot for {name}')

    imgOutput = os.path.join(outputDir, f"{name}.png")
    plt.savefig(imgOutput)

def main():
    folderPath = input("Enter the directory containing .csv files: ")
    outputPath = input("Enter the output directory for images: ")
    getPowerArray(folderPath, outputPath)

if __name__ == "__main__":
    main()