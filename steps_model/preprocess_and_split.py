# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Load data
# users_df = pd.read_csv('data/users.csv')
# products_df = pd.read_csv('data/products.csv')
# interactions_df = pd.read_csv('data/interactions.csv')

# # Encode categorical fields
# le_gender = LabelEncoder()
# le_location = LabelEncoder()
# le_category = LabelEncoder()

# users_df['gender_encoded'] = le_gender.fit_transform(users_df['gender'])
# users_df['location_encoded'] = le_location.fit_transform(users_df['location'])
# products_df['category_encoded'] = le_category.fit_transform(products_df['category'])

# # Save updated encoded files (opsional)
# users_df.to_csv('data/users.csv', index=False)
# products_df.to_csv('data/products.csv', index=False)

# # Train-test split
# train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)
# train_df.to_csv('data/train.csv', index=False)
# test_df.to_csv('data/test.csv', index=False)

# print("✅ Preprocessing done. Train & test data saved.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_and_split():
    # Load data
    users_df = pd.read_csv('data/users.csv')
    products_df = pd.read_csv('data/products.csv')
    interactions_df = pd.read_csv('data/interactions.csv')

    # Encode categorical fields
    le_gender = LabelEncoder()
    le_location = LabelEncoder()
    le_category = LabelEncoder()

    users_df['gender_encoded'] = le_gender.fit_transform(users_df['gender'])
    users_df['location_encoded'] = le_location.fit_transform(users_df['location'])
    products_df['category_encoded'] = le_category.fit_transform(products_df['category'])

    # Save updated encoded files
    users_df.to_csv('data/users.csv', index=False)
    products_df.to_csv('data/products.csv', index=False)

    # Train-test split
    train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print("✅ Preprocessing done. Train & test data saved.")
