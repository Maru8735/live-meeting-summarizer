import hashlib
import os
import pandas as pd

USER_DB = "users.csv"

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, hashed):
    """Check if a password matches its hash."""
    return hash_password(password) == hashed

def load_users():
    """Load users from the CSV database."""
    if not os.path.exists(USER_DB):
        return pd.DataFrame(columns=["username", "password", "email"])
    df = pd.read_csv(USER_DB)
    # Ensure email column exists for migration
    if 'email' not in df.columns:
        df['email'] = ""
    return df

def save_user(username, password, email):
    """Save a new user to the database."""
    users = load_users()
    if username in users['username'].values:
        return False, "Username already exists!"
    
    new_user = pd.DataFrame([[username, hash_password(password), email]], columns=["username", "password", "email"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DB, index=False)
    return True, "User registered successfully!"

def authenticate(username, password):
    """Verify user credentials."""
    users = load_users()
    if username not in users['username'].values:
        return False, "User not found!"
    
    stored_hash = users[users['username'] == username]['password'].values[0]
    if check_password(password, stored_hash):
        return True, "Login successful!"
    return False, "Invalid password!"
