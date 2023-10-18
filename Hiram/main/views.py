from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
import xml.etree.ElementTree as ET
import difflib
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import os
import urllib.parse
import requests

# Web page views
def home(request):
    return render(request,'home.html')


def aboutus(request):
    return render(request, 'about_us.html')

def methodology(request):
    return render(request, 'methodology.html')

def upload(request):
    return render(request, 'upload.html')



# Hiram view
from .Hiram import DrawIOObject, perform_clustering, FuzzyMatcher\

def list_files_in_folder(relative_folder_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, relative_folder_path)
    file_list = []
    # Ensure that the folder path exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Iterate through all files in the folder and its subdirectories
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    else:
        return []

    return file_list




ALL_PATHS = list_files_in_folder('./static/architecturs/backup') + list_files_in_folder('./static/architecturs/web') + list_files_in_folder('./static/architecturs/mobile') + list_files_in_folder('./static/architecturs/log_servers')

def receive_architecture_file(request):
    if request.method == 'POST' and request.FILES.get('xml_file'):
        xml_file = request.FILES['xml_file']
        test_object = DrawIOObject(xml_file)

        # Determine the closest match XML object using FuzzyMatcher
        best_score = 0
        best_match = ''
        response = ''
        for path in ALL_PATHS:
            web = DrawIOObject(path)
            matcher = FuzzyMatcher(test_object.flattened_matrix)
            if best_score < matcher.match(web.flattened_matrix):
                best_match = path
                best_score = matcher.match(web.flattened_matrix)

        # Create a response that includes both the family and the closest match
        response += f"The closest match XML object is {best_match}"

        best_match = best_match.replace(str(os.path.dirname(os.path.abspath(__file__))),"").replace("./","").replace("xml","png").split("/")[-1]
        # Return the response
        return render(request, 'display_closest_match.html', {'png_image': "images/" + best_match})

