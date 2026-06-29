import sqlite3
import datetime
import os

DB_PATH = "crm.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            customer_contact TEXT NOT NULL,
            car_of_interest TEXT NOT NULL,
            preferred_date_time TEXT NOT NULL,
            status TEXT DEFAULT 'New',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_messages (
            message_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_lead(customer_name: str, customer_contact: str, car_of_interest: str, preferred_date_time: str):
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO leads (customer_name, customer_contact, car_of_interest, preferred_date_time)
        VALUES (?, ?, ?, ?)
    ''', (customer_name, customer_contact, car_of_interest, preferred_date_time))
    conn.commit()
    conn.close()

def get_all_leads():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM leads ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_chat_message(phone_number: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_history (phone_number, role, content)
        VALUES (?, ?, ?)
    ''', (phone_number, role, content))
    conn.commit()
    conn.close()

def get_chat_history(phone_number: str, limit: int = 10):
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get last N messages in ascending order (chronological order)
    cursor.execute('''
        SELECT role, content FROM (
            SELECT role, content, created_at FROM chat_history 
            WHERE phone_number = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ) ORDER BY created_at ASC
    ''', (phone_number, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def is_message_processed(message_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM processed_messages WHERE message_id = ?', (message_id,))
    row = cursor.fetchone()
    conn.close()
    return row is not None

def mark_message_processed(message_id: str):
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO processed_messages (message_id) VALUES (?)', (message_id,))
    conn.commit()
    conn.close()
