
from multiprocessing import Process
import pickle
import curses  
import torchvision.transforms as transforms 
from torch.nn.functional import cosine_similarity, normalize
from facenet_pytorch import InceptionResnetV1
from bd_transac import get_all_employees, add_employee

resnet = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([
    transforms.ToTensor(),
])

def embedder(cropped_img):
    cropped_img = transform(cropped_img)
    emb = resnet(cropped_img.unsqueeze(0))
    return emb
def recognize_face(cord, threshold=0.6):
    embedding = embedder(cord)
    employees = get_all_employees()
    
    for employee in employees:
        db_embedding = pickle.loads(employee[3])
        similarity = calculate_similarity(embedding, db_embedding)
        if similarity > threshold:
            return employee[1]
    
    # Non-blocking user input using curses
    name = get_user_input()
    if name:
        add_employee(name, "programmer", embedding)
        return name
    return "Stranger"

def get_user_input():
    screen = curses.initscr()  # initialize curses screen
    curses.cbreak()  # disable line buffering
    screen.addstr(0, 0, "Please enter the name of the stranger: ")
    screen.refresh()
    name = screen.getstr()  # get user input without blocking
    curses.endwin()  # end curses
    return name.decode('utf-8').rstrip() 

def calculate_similarity(embedding1, embedding2):
    embedding1_normalized = normalize(embedding1, p=2, dim=1)
    embedding2_normalized = normalize(embedding2, p=2, dim=1)
    similarity = cosine_similarity(embedding1_normalized, embedding2_normalized)
    return similarity.item()
