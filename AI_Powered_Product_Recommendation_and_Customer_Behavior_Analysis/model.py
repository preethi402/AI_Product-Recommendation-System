import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load customer behavior data
customers = pd.read_csv("customer_data.csv")

# Create a customer-product rating matrix
rating_matrix = customers.pivot_table(
    index="Customer_ID",
    columns="Product_ID",
    values="Rating",
    fill_value=0
)

print("CUSTOMER-PRODUCT RATING MATRIX")
print(rating_matrix)

# Calculate similarity between customers
similarity = cosine_similarity(rating_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

print("\nCUSTOMER SIMILARITY")
print(similarity_df.round(2))
# Recommend products to a customer

customer_id = "C001"

# Get customers similar to the selected customer
similar_customers = similarity_df[customer_id].sort_values(
    ascending=False
)

# Remove the selected customer
similar_customers = similar_customers.drop(customer_id)

# Get products already purchased or rated by the customer
rated_products = rating_matrix.loc[customer_id]

# Find products the customer has not rated
unrated_products = rated_products[rated_products == 0].index

# Calculate recommendation scores
recommendations = {}

for product_id in unrated_products:

    score = 0

    for other_customer in similar_customers.index:

        rating = rating_matrix.loc[other_customer, product_id]

        similarity_score = similarity_df.loc[
            customer_id, other_customer
        ]

        score += rating * similarity_score

    recommendations[product_id] = score

# Sort recommendations
recommended_products = sorted(
    recommendations.items(),
    key=lambda x: x[1],
    reverse=True
)

# Display recommendations
products = pd.read_csv("products.csv")

print("\nRECOMMENDED PRODUCTS FOR", customer_id)

for product_id, score in recommended_products[:3]:

    product_name = products.loc[
        products["Product_ID"] == product_id,
        "Product_Name"
    ].values[0]

    print(product_name, "- Score:", round(score, 2))