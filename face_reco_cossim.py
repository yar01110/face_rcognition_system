                                                                 
import pickle
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
from bd_transac import get_all_employees
from facenet_pytorch import InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([
    transforms.ToTensor(),
])
def embeder(croped_img):
    croped_img = transform(croped_img)
    emb = resnet(croped_img.unsqueeze(0))
    return embeder
def recognize_face(cord, threshold=0.6, c=None):
    embedding=embeder(cord)
    employees = get_all_employees()
    
    for employee in employees:
        db_embedding = pickle.loads(employee[3])
        similarity = calculate_similarity(embedding, db_embedding)
        if similarity > threshold:
            return employee[2]  
    return "Stranger"




def calculate_similarity(embedding1, embedding2):
    embedding1_normalized = torch.nn.functional.normalize(embedding1, p=2, dim=1)
    embedding2_normalized = torch.nn.functional.normalize(embedding2, p=2, dim=1)
    similarity = cosine_similarity(embedding1_normalized, embedding2_normalized)
    return similarity.item()


