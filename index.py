from constants import EMB_COL, ID_COL
import numpy as np
import faiss


def create_index_from_df(df, idx_mapping=None):
    """
    Create a Faiss index that uses cosine similarity. Vectors from a given pandas dataframe are added to the index

    :param df: pandas DataFrame containing the embeddings to add to the index. Assumes the embeddings are under column `emb` and ids are under column `img_id`
    :param idx_mapping: dict mapping id in the Faiss index to id in the pandas DataFrame. If None, the mapping will be created by iterating through the rows of the `img_id` column

    :return: 2-len tuple of (Faiss index, idx_mapping)
    """
    if idx_mapping is None:
        idx_mapping = {idx: img_id for idx, img_id in enumerate(df[ID_COL])}

    vectors = np.array(df[EMB_COL].to_list())
    # need vectors as type float32 in order to use faiss.normalize_L2
    # ref: https://stackoverflow.com/a/76037727
    vectors = vectors.astype(np.float32)

    # normalize vectors in order to use cosine similarity with IP index
    faiss.normalize_L2(vectors)
    print("Creating index with vectors of shape:", vectors.shape)

    d = vectors.shape[1]  # dimension of vectors
    index = faiss.IndexFlatIP(d)
    # Add vectors to the index
    index.add(vectors)

    return index, idx_mapping
