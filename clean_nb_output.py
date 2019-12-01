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
    removeComments('03_28_resnet50_bgconst_valid_pewter_mixup_20191030.py','03_28_resnet50_bgconst_valid_pewter_mixup_20191030_clean.py')
