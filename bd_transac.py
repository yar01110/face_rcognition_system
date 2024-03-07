import sqlite3
import pickle
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



    





