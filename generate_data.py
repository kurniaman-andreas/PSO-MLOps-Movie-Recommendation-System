import pandas as pd
import numpy as np

# --- Generate synthetic data ---
num_users = 1000
num_products = 500

# Users
users = {
    'user_id': np.arange(1, num_users + 1),
    'age': np.random.randint(18, 70, size=num_users),
    'gender': np.random.choice(['M', 'F'], size=num_users),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_users)
}
users_df = pd.DataFrame(users)
users_df.to_csv('data/users.csv', index=False)

# Products
products = {
    'product_id': np.arange(1, num_products + 1),
    'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], size=num_products),
    'price': np.round(np.random.uniform(5, 500, size=num_products), 2),
    'rating': np.round(np.random.uniform(1, 5, size=num_products), 1)
}
products_df = pd.DataFrame(products)
products_df.to_csv('data/products.csv', index=False)

# Interactions
interactions = {
    'user_id': np.random.choice(users_df['user_id'], size=5000),
    'product_id': np.random.choice(products_df['product_id'], size=5000),
    'rating': np.random.randint(1, 6, size=5000),
    'timestamp': pd.date_range(start='2023-01-01', periods=5000, freq='T')
}
interactions_df = pd.DataFrame(interactions)
interactions_df.to_csv('data/interactions.csv', index=False)

print("✅ Data synthetic saved to /data/")
