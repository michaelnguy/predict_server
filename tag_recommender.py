import numpy as np
import faiss

from collections import defaultdict

class TagRecommender:
    """
    Class to perform tag recommendation for images
    """

    def __init__(self, index, idx_mapping, img_tag_list,
                num_rec_tags=5, weight_tags=True):
        """
        Initialize the TagRecommender

        :param index: Faiss index to be used for finding similar images
        :param idx_mapping: mapping for vector id in the Faiss index to id in the image dataset
        :param img_tag_list: list of lists of ground truth tags for the images in the Faiss index. Assumes that img_tag_list[i] is the list of ground truth tags for image i
        :param img_vectorizer: model used to generate embeddings or vectors for an image. The model should take in an image and return a numpy array
        :param num_rec_tags: the number of desired tags to be recommended for a given image
        :param weight_tags: whether to weight the recommended tags by Faiss score
        """
        self.index = index
        self.vector_id_to_img_id_mapping = idx_mapping
        self.img_tag_list = img_tag_list
        self.default_num_rec_tags = num_rec_tags
        self.weight_tags = weight_tags

    def get_tags_for_image(self, img, img_vectorizer=None, num_tags=None):
        assert img_vectorizer is not None, "No model provided to vectorize images. Please provide an image vectorizer or use get_tags_for_image_vector instead."
        if type(img) is not list:
            img = [img]
        img_vector = img_vectorizer(img)
        return self.get_tags_for_image_vector(img_vector, num_tags)

    def get_tags_for_image_vector(self, img_vector, num_tags=None):
        if num_tags is None:
            num_tags = self.default_num_rec_tags
        img_vector = img_vector.astype(np.float32)
        faiss.normalize_L2(img_vector)
        if len(img_vector.shape) < 2:
            print("Getting tags for single image. Adding dimension to vector. Resulting vector shape:", img_vector.shape)
            img_vector = np.expand_dims(img_vector, axis=0)

        faiss_scores, faiss_idx = self._search_index(img_vector, num_tags)

        rec_tags_w_weights = self._get_rec_tags_and_weights(faiss_scores, faiss_idx)

        rec_tags = [list(rec_tags_dict.keys())[:num_tags] for rec_tags_dict in rec_tags_w_weights]

        return rec_tags

    def _search_index(self, query_vector, num_neighbors):
        scores, indices = self.index.search(query_vector, k=num_neighbors)
        return scores,indices

    def _get_rec_tags_and_weights(self, scores, indices):
        rec_tags_list = []

        # iterate through each image we want to recommend tags for
        for score_list_i,idx_list_i in zip(scores, indices):
            # create new counter of rec tags per image
            recommended_tags_i = defaultdict(lambda : 0)

            # iterate through the similar images
            for score,idx in zip(score_list_i, idx_list_i):
                sim_img_id = self.vector_id_to_img_id_mapping.get(idx)
                if sim_img_id is None:
                    print("Unable to find image id for vector id:", idx)
                    continue
                sim_img_tags = self.img_tag_list[sim_img_id]

                for tag in sim_img_tags:
                    if self.weight_tags:
                        recommended_tags_i[tag] += score
                    else:
                        recommended_tags_i[tag] += 1

            # sort recommended_tags_i dict by weight
            recommended_tags_i = dict(sorted(recommended_tags_i.items(), key=lambda t:t[1], reverse=True))
            rec_tags_list.append(recommended_tags_i)

        return rec_tags_list