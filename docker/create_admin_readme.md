# Admin Account Setup Guide

This guide explains how to create an initial **Admin user** for DataBotti.

## 1. Generate a Password Hash

Open a Python shell and run:

```python
from werkzeug.security import generate_password_hash
print(generate_password_hash("ChangeMe123!"))
```

Replace `"ChangeMe123!"` with your desired password.  
Copy the resulting hash string.

## 2. Insert Admin User into the Database

In your database (e.g., via `mysql` client or init script), run the following SQL:

```sql
INSERT INTO users (email, password_hash, username, is_admin)
VALUES ('admin@example.com', '<PASTE_HASH_HERE>', 'admin', 1);
```

- Replace `admin@example.com` with the desired email.
- Replace `<PASTE_HASH_HERE>` with the hash you generated in step 1.
- `is_admin` is set to `1` to mark this user as an Admin.

## 3. Login via Web

Once the user exists in the database, you can log in with the email and the plain password you chose in step 1.

---

✅ You now have an Admin account ready to use in DataBotti.
