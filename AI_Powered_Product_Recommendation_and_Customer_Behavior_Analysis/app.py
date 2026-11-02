import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

# Load datasets
products = pd.read_csv("products.csv")
customers = pd.read_csv("customer_data.csv")

# Create customer-product rating matrix
rating_matrix = customers.pivot_table(
    index="Customer_ID",
    columns="Product_ID",
    values="Rating",
    fill_value=0
)

# Calculate customer similarity
similarity = cosine_similarity(rating_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

# Recommendation function
def recommend_products(customer_id):

    similar_customers = similarity_df[customer_id].sort_values(
        ascending=False
    ).drop(customer_id)

    rated_products = rating_matrix.loc[customer_id]

    unrated_products = rated_products[
        rated_products == 0
    ].index

    recommendations = {}

    for product_id in unrated_products:

        score = 0

        for other_customer in similar_customers.index:

            rating = rating_matrix.loc[
                other_customer, product_id
            ]

            similarity_score = similarity_df.loc[
                customer_id, other_customer
            ]

            score += rating * similarity_score

        recommendations[product_id] = score

    recommended_products = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return recommended_products


# Website title
st.title("🛍️ AI Product Recommendation System")

st.write(
    "Discover products recommended based on customer preferences."
)

# Display products
st.subheader("Available Products")

st.dataframe(products, use_container_width=True)

# Customer selection
customer_id = st.selectbox(
    "Select Customer",
    customers["Customer_ID"].unique()
)

# Customer activity
st.subheader("Customer Activity")

customer_activity = customers[
    customers["Customer_ID"] == customer_id
]

st.dataframe(customer_activity, use_container_width=True)

# Recommendations
st.subheader("Recommended Products")

recommended_products = recommend_products(customer_id)

if recommended_products:

    for product_id, score in recommended_products[:3]:

        product = products[
            products["Product_ID"] == product_id
        ]

        if not product.empty:

            product_name = product.iloc[0]["Product_Name"]
            price = product.iloc[0]["Price"]

            st.success(
                f"{product_name} | Price: ₹{price} | "
                f"Recommendation Score: {score:.2f}"
            )

else:

    st.info("No new recommendations available.")
   # Customer Analytics Dashboard

st.title("Customer Analytics Dashboard")

# Display summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", customers["Customer_ID"].nunique())

with col2:
    st.metric("Total Products", len(products))

with col3:
    st.metric("Total Interactions", len(customers))

# Most popular products
st.subheader("Most Popular Products")

popular_products = customers["Product_ID"].value_counts().reset_index()

popular_products.columns = ["Product_ID", "Total Ratings"]

popular_products = popular_products.merge(
    products,
    on="Product_ID"
)

st.bar_chart(
    popular_products.set_index("Product_Name")["Total Ratings"]
)

# Customer rating analysis
st.subheader("Customer Rating Analysis")

st.bar_chart(
    customers.groupby("Customer_ID")["Rating"].mean()
) 
