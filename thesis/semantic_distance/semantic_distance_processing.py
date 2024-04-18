# imports from __init__.py
from . import PCA  # Import the PCA class
from . import np  # Import numpy
from . import torch  # Import PyTorch
from . import pd  # Import pandas
from . import cosine_similarity # Import sklearn function for cosine similarity

model_dim = 768  # Number of model dimensions (768 for BERT base)

def process_embeddings(version_name, num_pcs=0.95, pca=True, save=True):
    """
    This function takes BERT model embeddings as input, 
    averages them by segment across the sequence length dimension, 
    and performs principal component analysis.
    
    :param version_name: Points to unprocessed model embeddings
    :param num_pcs: Number of principal components
    :param pca: Whether to apply PCA
    :param save: Whether to save output
    :return: Processed embedding segments (after PCA if applicable)
    """

    # Load tensor with embeddings
    embeddings = torch.load(f"data/{version_name}.pt", map_location=torch.device('cpu'))

    # Determine number of segments in tensor
    num_segments = len(embeddings)

    # Initialize list to store flattened segment embeddings
    segment_list = []

    # Loop over each segment
    for i in range(num_segments):

        # Average embeddings across sequence length dimension
        this_segment = embeddings[i].mean(dim=1).flatten()

        # Flatten and append to the list
        segment_list.append(np.array(this_segment.reshape(1, -1)))

    # Stack segments into a matrix
    processed_segments = np.column_stack((segment_list)).reshape(num_segments*1, model_dim)

    # If applying PCA
    if pca:
        processed_segments = apply_pca(processed_segments, version_name, num_pcs, save)
    else:
        # Save processed segments if true
        if save:
            save_output(processed_segments, "processed_segments", version_name)

    return processed_segments

def apply_pca(embeddings, version_name, num_pcs=0.95, save = True):
    """
    Function to reduce the dimensionality of model embeddings 
    by applying principal component analysis.
    
    :param embeddings: Embeddings averaged over the sequence length dimension
    :param num_pcs: Number of principal components (preserve 95% variance by default)
    :param save: Whether to save output
    :return: PCA-transformed embeddings
    """

    # Create PCA object with the specified number of principal components
    pca_object = PCA(n_components=num_pcs)

    # Apply PCA to reduce the dimensionality of the segments
    pca_embeddings = pca_object.fit_transform(embeddings)
    
    # Save processed segments if true, denoting that PCA was applied
    if save:
        save_output(pca_embeddings, "processed_segments", (version_name + "_pca"))
                    
    return pca_embeddings

def cosine_distance(vector1, vector2, version_name, pca=True, save=True):
    """
    Calculate cosine distances between two vectors.
    
    :param vector1: First vector
    :param vector2: Second vector
    :param version_name: Version name (used for saving)
    :param pca: Whether PCA was applied to embeddings
    :param save: Whether to save output
    :return: List of cosine distances
    """

    # Determine number of segments in tensor
    num_segments = 2700
                    
    # Create list to collect distance scores
    distance_scores = []

    # Compute cosine distance between each pair of segments for each in list
    for i in range(num_segments):
        
        # First, compute cosine similarity between pair
        cosine_similarities = cosine_similarity(vector1[i].reshape(1, -1), vector2[i].reshape(1, -1))
        
        # Compute cosine distance from similarity
        cosine_scores = 1 - cosine_similarities
                    
        # Append score for segment to list
        distance_scores.append(cosine_scores[0][0])

    # Save cosine distance scores if called
    if save:
                    
        # denote whether PCA was applied before computing cosine distance
        if pca == True:
            save_output(distance_scores, "cosine_distance", (version_name + "_pca"))
        else:
            save_output(distance_scores, "cosine_distance", version_name)

    return distance_scores

def save_output(output_data, cat, version_name):
    """
    Save the output data to a CSV file.
    
    :param output_data: Processed data to be saved
    :param cat: Category for output file
    :param version_name: Version name (used for saving)
    :return: None
    """

    df = pd.DataFrame(output_data)
    df.to_csv(f"output/{cat}/{version_name}.csv", index=False, header=False)

    return