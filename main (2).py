from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

TOKEN: Final = "" # Here add the token
BOT_USERNAME : Final = "@travelhelpIS_bot"

# Define dataset path
dataset_path = r'C:\Users\amans\Desktop\UNI\Intelligence systems\Project Telegram Bot\Dataset.csv'

# Define log and graph paths dynamically based on the dataset path
log_path = os.path.join(os.path.dirname(dataset_path), 'user_queries_log.csv')
graph_path = os.path.join(os.path.dirname(dataset_path), 'clustering_graph.png')

# Check if the dataset file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

# Load the dataset
dataset = pd.read_csv(dataset_path)

# Sample data for initial clustering
sample_data = [
    "I want a cheap location",
    "I want an expensive location",
    "I want a medium priced location",
    "I want a location with a great view",
    "Suggest a cheap place",
    "Show me an expensive place",
    "Any medium-priced suggestions?",
    "Looking for a place with a nice view"
]

# Load or initialize the log file
if os.path.exists(log_path):
    user_log_df = pd.read_csv(log_path)
else:
    user_log_df = pd.DataFrame(columns=["query", "cluster"])

# Combine sample data and user log data for training
def get_combined_data():
    combined_data = sample_data + user_log_df["query"].tolist()
    return combined_data

# Train the clustering model
def train_clustering_model():
    combined_data = get_combined_data()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(combined_data)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(kmeans, 'kmeans.joblib')

    # Convert the sparse matrix to dense and then to a numpy array for plotting
    X_dense = X.todense()
    X_array = np.asarray(X_dense)

    # Reduce dimensionality for plotting
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_array)
    
    plot_clusters(X_reduced, kmeans)

# Load the trained model and vectorizer
def load_clustering_model():
    vectorizer = joblib.load('vectorizer.joblib')
    kmeans = joblib.load('kmeans.joblib')
    return vectorizer, kmeans

# Plot the clusters and save to a file with labels and legend
def plot_clusters(X, kmeans):
    plt.figure(figsize=(10, 8))
    colors = ['yellow', 'blue', 'green', 'red']
    labels = ['cheap', 'expensive', 'medium', 'view']
    
    for i, color in enumerate(colors):
        plt.scatter(X[kmeans.labels_ == i, 0], X[kmeans.labels_ == i, 1], c=color, label=labels[i])
    
    plt.title("Clustering of User Queries")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig(graph_path)
    plt.close()

# Start command
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! Depending on what you're looking for, you can ask for a cheap, expensive, medium-priced location! "
        "Or you could simply ask for a location with a great view."
    )

# Help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! Thank you for using us, your friendly telegram bot.\n"
        "Our goal is to assist you to find your desirable vacation spot in KL."
        "You can ask for a type of location by saying things like:\n"
        "- I want a cheap location\n"
        "- I want an expensive location\n"
        "- I want a medium priced location\n"
        "- I want a location with a great view\n"
        "Or, you could simply type in the keywords, such as expensive, cheap, medium or view.\n"
        "In short, once you request for your type, we will look through our data and make sure to match with your needs.\n"
        "Thank you!"
    )

# Log user queries
def log_user_query(text: str, cluster: int):
    log_entry = pd.DataFrame({"query": [text], "cluster": [cluster]})
    if not os.path.exists(log_path):
        log_entry.to_csv(log_path, index=False)
    else:
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, log_entry], ignore_index=True)
        log_df.to_csv(log_path, index=False)

# Generate a response based on user input and clusters
def handle_response(text: str) -> str:
    vectorizer, kmeans = load_clustering_model()
    
    text_vectorized = vectorizer.transform([text])
    cluster = kmeans.predict(text_vectorized)[0]
    
    # Log the user query and cluster
    log_user_query(text, cluster)
    
    response = ""

    # Predefined responses
    if 'cheap' in text.lower():
        response = get_location_response('cheap')
    elif 'expensive' in text.lower():
        response = get_location_response('expensive')
    elif 'medium' in text.lower():
        response = get_location_response('medium')
    elif 'view' in text.lower():
        response = get_location_response('view')
    else:
        response = (
            "I do not understand what you typed. Please try again or type /help for guidance. "
            "If you are looking for specific types of locations, you can use keywords like 'cheap', 'expensive', "
            "'medium', or 'view' in your query."
        )

    return response

def get_location_response(location_type: str) -> str:
    if 'Type' not in dataset.columns:
        return "The dataset does not contain the required 'Type' column."
    
    filtered_locations = dataset[dataset['Type'].str.lower() == location_type]
    if not filtered_locations.empty:
        location = filtered_locations.sample().iloc[0]
        return (
            f"Okay, your destination will be {location['Tourist Attractions']} at {location['Location']}.\n"
            f"{location['Activity']}\n"
            f"The price is {location['Ticket price(per person)']} per person."
        )
    else:
        return f"Sorry, I couldn't find any {location_type} locations."

# Handle messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text: str = update.message.text
    response: str = handle_response(text)
    await update.message.reply_text(response)
    train_clustering_model()

# Handle errors
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass

if __name__ == "__main__":
    train_clustering_model()

    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    app.run_polling(poll_interval=3)
