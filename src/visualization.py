
import matplotlib.pyplot as plt
import seaborn as sns
import umap

class NarrativeVisualizer:
    def __init__(self, config):
        self.config = config

    def create_narrative_space_visualization(self, embeddings, labels):
        reducer = umap.UMAP(n_neighbors=self.config['visualization']['umap_n_neighbors'],
                              min_dist=self.config['visualization']['umap_min_dist'],
                              n_components=self.config['visualization']['umap_n_components'],
                              metric=self.config['visualization']['umap_metric'])
        
        embedding_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels, palette=sns.color_palette("hsv", len(np.unique(labels))))
        plt.title('Narrative Space Visualization')
        plt.show()
