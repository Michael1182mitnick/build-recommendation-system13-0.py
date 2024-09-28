# Build Recommendation System
# Implement a recommendation system using collaborative filtering, content-based filtering, or a hybrid approach to suggest products or content based on user behavior.
# pip install scikit-surprise
# Collaborative Filtering with SVD

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import cross_validate

# Load a dataset
# Assuming you have a user-movie rating dataset in CSV format, with columns 'userId', 'movieId', 'rating'
# You can also load your own dataset with a similar format.
data = Dataset.load_builtin('ml-100k')

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Use the SVD algorithm
model = SVD()

# Train the model on the training data
model.fit(trainset)

# Test the model on the test data
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Function to get recommendations for a user


def get_movie_recommendations(user_id, model, data, n_recommendations=5):
    # Get the list of all movies
    movie_ids = data.build_full_trainset().all_items()
    movie_ids = [data.to_raw_iid(i) for i in movie_ids]

    # Predict ratings for all movies not yet rated by the user
    user_ratings = []
    for movie_id in movie_ids:
        user_ratings.append((movie_id, model.predict(user_id, movie_id).est))

    # Sort by rating and return top N recommendations
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    return user_ratings[:n_recommendations]


# Example: Get top 5 recommendations for user with id 1
recommendations = get_movie_recommendations(1, model, data)
print(recommendations)
