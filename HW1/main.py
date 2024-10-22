import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file) 
        next(file)
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def load_visim_400(file_path):
    word_pairs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word1 = parts[0]
            word2 = parts[1]
            word_pairs.append((word1, word2))
    return word_pairs

def load_data_from_file(file_path, embeddings):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split(' ')
            if len(values) < 2:
                continue
            
            if values[0] in embeddings and values[1] in embeddings:
                data.append(cosine_similarity(embeddings[values[0]], embeddings[values[1]]))
    return data

def load_data(syn_data_path, ant_data_path, embeddings):
    syn_data = load_data_from_file(syn_data_path, embeddings)
    ant_data = load_data_from_file(ant_data_path, embeddings)

    data = syn_data + ant_data
    labels = [1] * len(syn_data) + [-1] * len(ant_data)

    return np.array(data).reshape(-1, 1), np.array(labels)

def load_test_syn_ant_data(file_path, embeddings):
    data = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            parts = line.strip().split()
            
            if parts[0] in embeddings and parts[1] in embeddings:
                data.append(cosine_similarity(embeddings[parts[0]], embeddings[parts[1]]))
                labels.append(1 if parts[2] == 'SYN' else -1) 

    return np.array(data).reshape(-1, 1), np.array(labels)

# Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def calculate_similarity_for_word_pairs(embeddings, word_pairs):
    results = []
    for word1, word2 in word_pairs:
        if word1 in embeddings and word2 in embeddings:
            similarity = cosine_similarity(embeddings[word1], embeddings[word2])
            results.append((word1, word2, similarity))
        else:
            results.append((word1, word2, None)) 
    return results

def find_k_nearest_words(word, embeddings, k):
    if word not in embeddings or k <= 0:
        return []

    word_vector = embeddings[word]
    similarities = []

    for other_word, other_vector in embeddings.items():
        if other_word != word: 
            similarity = cosine_similarity(word_vector, other_vector)
            similarities.append((other_word, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def train_classifier(data, labels):
    model = LogisticRegression()
    model.fit(data, labels)
    return model

############################################### TEST ########################################################
def test_calculate_similarity_for_word_pairs():
    embedding_file = 'model/W2V_150.txt'  
    visim_file = 'data/Visim-400.txt'          
    
    embeddings = load_embeddings(embedding_file)
    word_pairs = load_visim_400(visim_file)
    
    similarities = calculate_similarity_for_word_pairs(embeddings, word_pairs)
    
    for word1, word2, sim in similarities:
        if sim is not None:
            print(f"Cosine similarity between '{word1}' and '{word2}' is: {sim:.4f}")
        else:
            print(f"Embedding for '{word1}' or '{word2}' not found.")

def test_find_k_nearest_words():
    embedding_file = 'model/W2V_150.txt'  
    embeddings = load_embeddings(embedding_file)
    
    word = "cảm_giác"  
    k = 5  
    
    nearest_words = find_k_nearest_words(word, embeddings, k)
    print(f"Top {k} words nearest to '{word}':")
    for other_word, similarity in nearest_words:
        print(f"{other_word}: {similarity:.4f}")

def test_syn_ant_classification():
    embedding_file = 'model/W2V_150.txt'    
    syn_data_path = "data/Synonym_vietnamese.txt"  
    ant_data_path = "data/Antonym_vietnamese.txt"
    test_path = "data/400_noun_pairs.txt"
    
    embeddings = load_embeddings(embedding_file)
    data, labels = load_data(syn_data_path, ant_data_path, embeddings)
    X_test, y_test = load_test_syn_ant_data(test_path, embeddings)
    
    model = train_classifier(data, labels)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    # test_calculate_similarity_for_word_pairs()  # 1
    # test_find_k_nearest_words()  # 2
    test_syn_ant_classification()  # 3
