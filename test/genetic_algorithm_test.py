from utils.network_utils import *
from tensorflow.keras.models import load_model

# Initial models from dataset
modelFW_1_0 = load_model("../../StartingNets/modelFW_1_0")
modelTW_1_0 = load_model("../../StartingNets/modelTW_1_0")
modelFB_1_0 = load_model("../../StartingNets/modelFB_1_0")
modelTB_1_0 = load_model("../../StartingNets/modelTB_1_0")

modelW_1_0 = (modelFW_1_0, modelTW_1_0)
modelB_1_0 = (modelFB_1_0, modelTB_1_0)

initial_models_white = [modelW_1_0]
initial_models_black = [modelB_1_0]

# Generate new models based on the first ones
net_num = 40
gen_num = 0
new_models_white = generate(initial_models_white, net_num)
new_models_black = generate(initial_models_black, net_num)

path = "../../NeuroEvolution/0/"
save_models(new_models_white, path, "W", gen_num)
save_models(new_models_black, path, "B", gen_num)
