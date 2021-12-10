import os

def dance_style(path):
    filename = path.name
    arr = filename.split("_")
    return [arr[1]]

def annotation(path):
    path = str(path)+".annotation.txt"
    annotation = open(path, "r").read()
    return annotation.split(" ")

