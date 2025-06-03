import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import pickle

# Kalau perlu install library jalankan perintah di terminal:
# pip install pandas numpy scikit-surprise scikit-learn

# --- Generate synthetic data ---
num_users = 1000
num_products = 500

user_data = {
    'user_id': np.arange(1, num_users + 1),
    'age': np.random.randint(18, 70, size=num_users),
    'gender': np.random.choice(['M', 'F'], size=num_users),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_users)
}
users_df = pd.DataFrame(user_data)

product_data = {
    'product_id': np.arange(1, num_products + 1),
    'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], size=num_products),
    'price': np.round(np.random.uniform(5, 500, size=num_products), 2),
    'rating': np.round(np.random.uniform(1, 5, size=num_products), 1)
}
products_df = pd.DataFrame(product_data)

interaction_data = {
    'user_id': np.random.choice(users_df['user_id'], size=5000),
    'product_id': np.random.choice(products_df['product_id'], size=5000),
    'rating': np.random.randint(1, 6, size=5000),
    'timestamp': pd.date_range(start='2023-01-01', periods=5000, freq='T')
}
interactions_df = pd.DataFrame(interaction_data)

# --- Preprocessing ---

# Encode categorical variables
label_encoder = LabelEncoder()
users_df['gender_encoded'] = label_encoder.fit_transform(users_df['gender'])
users_df['location_encoded'] = label_encoder.fit_transform(users_df['location'])
products_df['category_encoded'] = label_encoder.fit_transform(products_df['category'])

# Create user-product matrix
user_product_matrix = interactions_df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# Train-test split interaction data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions_df[['user_id', 'product_id', 'rating']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Evaluate model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Save model
model_filename = 'model/svd_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_filename}")
