import sqlite3
import pickle
from PIL import Image
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])
from facenet_pytorch import InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval()
def deco(func):
    
    conn = sqlite3.connect('employees.db')
    c = conn.cursor()

    def wrapper(*args, **kwargs):
            result = func(*args, **kwargs, c=c) 
            conn.commit()
            return result
        

    return wrapper


#c.execute('''CREATE TABLE IF NOT EXISTS employees
#             (id INTEGER PRIMARY KEY AUTOINCREMENT,
#              name TEXT NOT NULL,
#              role TEXT NOT NULL,
#              embedding BLOB NOT NULL)''')
@deco
def add_employee(name, role, embedding, c=None):  
    embedding = pickle.dumps(embedding)
    c.execute("INSERT INTO employees (name, role, embedding) VALUES (?, ?, ?)", (name, role, sqlite3.Binary(embedding)))

@deco
def get_all_employees( c=None):
    c.execute("SELECT * FROM employees")
    return c.fetchall()

@deco
def store_embedding_in_database(embedding,c=None):
    
    embedding_pickle = pickle.dumps(embedding)
    c.execute("INSERT INTO employees (embedding) VALUES (?)", (sqlite3.Binary(embedding_pickle),))

@deco
def delete_employee(employee_id, c=None):
    c.execute("DELETE FROM employees WHERE id = ?", (employee_id,))

for i in range(2,8):
    delete_employee(i)
#delete_employee(3)



