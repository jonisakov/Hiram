
from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
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


urlpatterns = [
    # Web pages
    path('admin/', admin.site.urls),
    path('', views.home, name="home"),
    path('about_us/',views.aboutus, name="about_us"),
    path('Methodology/', views.methodology, name="methodology"),
    path('upload/',views.upload, name="upload"),

    # Hiram views
    path('receive_architecture_file/', views.receive_architecture_file, name='receive_architecture_file'),

]
