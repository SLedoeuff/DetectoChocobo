#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Créer par Sacha Le Doeuff

from time import time
from PIL import Image
from resizeimage import resizeimage
import glob
import random

"""-------------------------------------------------------
#             Resize des images + Rotation               # 
-------------------------------------------------------"""

inc = 1

targetOui = "OuiChoco/"
targetNon = "NonChoco/"

filesOui = glob.glob('Oui/*.jpg')
filesNon = glob.glob('Non/*.jpg')

#image de chocobo
for file in filesOui:
	with open(file, 'r+b') as f:
		with Image.open(f) as image:
			image = image.resize((128,128), Image.ANTIALIAS)
			image.save(targetOui+str(inc)+"_base.jpg", image.format)
			
			image = image.rotate(90,expand=1)
			image.save(targetOui+str(inc)+"_90.jpg",image.format)
			
			image = image.rotate(90,expand=1)
			image.save(targetOui+str(inc)+"_180.jpg",image.format)
			
			image = image.rotate(90,expand=1)
			image.save(targetOui+str(inc)+"_270.jpg",image.format)

			inc += 1


#image d'autres choses que des chocobos
for file in filesNon:
	with open(file, 'r+b') as f:
		with Image.open(f) as image:
			image = image.resize((128,128), Image.ANTIALIAS)
			image.save(targetNon+str(inc)+"_base.jpg", image.format)
			
			image = image.rotate(90,expand=1)
			image.save(targetNon+str(inc)+"_90.jpg",image.format)
			
			image = image.rotate(90,expand=1)
			image.save(targetNon+str(inc)+"_180.jpg",image.format)
			
			image = image.rotate(90,expand=1)
			image.save(targetNon+str(inc)+"_270.jpg",image.format)

			inc += 1

