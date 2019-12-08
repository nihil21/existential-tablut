from utils.network_utils import *
from keras.models import load_model

modelFB = load_model("modelFB")
modelTB = load_model("modelTB")

modelFW = load_model("modelFW")
modelTW = load_model("modelTW")

modelsB = generate([[modelFB, modelTB]], 2)

save_models(modelsB, "/media/tree/Dati/GIACOMO/Unibo/Magistrale/Ianno/Challange Tablut/Code/nets/prova/", "B", "0")