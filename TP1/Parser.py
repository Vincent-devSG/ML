#!usr/bin/python

from optparse import OptionParser


def main():
    parser = OptionParser(
        "python3 split.py [arguments] [file ..]      split file in two new files depending on the split percentage")
    parser.add_option("-f", dest="file", help="File name")
    parser.add_option("-s", dest="split", help="Split percentage [0-100]")

    (options, args) = parser.parse_args()

    if (not options.file):
        parser.print_help()
        print("")
        exit(0)
    else:
        file_ = options.file
        split_ = 50
        if (options.split):
            split_ = int(options.split)
        split(file_, split_, parser)


def split(file, split, parser):
    try:
        inputFile = open(file, "r")
        data = inputFile.read().splitlines()
        inputFile.close()

        dataOutput = ['', '']
        turningIndex = (split / 100) * len(data)
        iteration = 0

        for line in data:
            dataOutput[int(iteration / turningIndex)] += line + "\n"
            iteration += 1

        dataOutput[0] = dataOutput[0][:dataOutput[0].rfind('\n')]
        dataOutput[1] = dataOutput[1][:dataOutput[1].rfind('\n')]

        outputFile = open(file.split(".")[0] + "_" + str(split) + "." + file.split(".")[1], "w")
        outputFile.write(dataOutput[0])
        outputFile.close()

        string = ""
        if (split == 50):
            string = "_2"

        outputFile = open(file.split(".")[0] + "_" + str(100 - split) + string + "." + file.split(".")[1], "w")
        outputFile.write(dataOutput[1])
        outputFile.close()

        print("File successfully splitted...")
    except:
        parser.print_help()
        print("")
        exit(0)
        pass


main()