def removeComments(inputFileName, outputFileName):

    input = open(inputFileName, "r")
    output = open(outputFileName, "w")

    output.write(input.readline())

    for line in input:
        if not line.lstrip().startswith("# In["):
            output.write(line)

    input.close()
    output.close()

if __name__ == "__main__":
    removeComments('01_data_prep_rasterio.py','01_data_prep_rasterio_clean.py')
