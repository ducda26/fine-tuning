from lib import *
from config import *
from utils import load_model
from image_transform import ImageTransform

class_index = ["ants", "bees"]

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index
    def predict_max(self, output): # [0.9, 0.1]
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.class_index[max_id]
        return predicted_label
       
predictor = Predictor(class_index)

def predict(img):
    #Prepare network
    use_pretrained = True
    net = models.vgg16(pretrained = use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.eval()
    
    #prepare model
    model = load_model(net, save_path)
    
    #Prepare input_img
    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase="val")
    img = img.unsqueeze_(0) #(channel, height, width) --> (1, channel, height, width)
    
    #Predict
    output = model(img)
    response = predictor.predict_max(output)
    
    return response