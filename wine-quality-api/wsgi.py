import sys
import os

path = '/home/TU-USUARIO/DEPLOYML/wine-quality-api'
if path not in sys.path:
    sys.path.append(path)

os.chdir(path)

from app import app as application

print("✅ WSGI configurado correctamente")