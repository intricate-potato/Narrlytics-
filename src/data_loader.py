import os
import pandas as pd
from scipy.sparse import csr_matrix
import yaml
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

class MINDDataLoader:
    def __init__(self, config_path, data_dir, version='small', download=True):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = data_dir
        self.version = version
        self.train_path = os.path.join(self.data_dir, self.version, 'train')
        self.dev_path = os.path.join(self.data_dir, self.version, 'dev')

        if download:
            self.download_mind_dataset()
            
        self.news_data = self.load_news_data()
        self.behaviors_data = self.load_behaviors_data()

    def download_mind_dataset(self):
        mind_url, mind_train_dataset, mind_dev_dataset, _ = get_mind_data_set(self.version)

        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
            download_deeprec_resources(mind_url, self.train_path, mind_train_dataset)

        if not os.path.exists(self.dev_path):
            os.makedirs(self.dev_path)
            download_deeprec_resources(mind_url, self.dev_path, mind_dev_dataset)

    def load_news_data(self):
        train_news_path = os.path.join(self.train_path, 'news.tsv')
        dev_news_path = os.path.join(self.dev_path, 'news.tsv')
        
        train_news_df = pd.read_csv(train_news_path, sep='\t', header=None, names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities'])
        dev_news_df = pd.read_csv(dev_news_path, sep='\t', header=None, names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities'])
        
        news_df = pd.concat([train_news_df, dev_news_df]).drop_duplicates(subset=['NewsID'])

        if self.config['data']['combine_title_abstract']:
            news_df['Text'] = news_df['Title'].fillna('') + ' ' + news_df['Abstract'].fillna('')
        
        return news_df

    def load_behaviors_data(self):
        train_behaviors_path = os.path.join(self.train_path, 'behaviors.tsv')
        dev_behaviors_path = os.path.join(self.dev_path, 'behaviors.tsv')

        train_behaviors_df = pd.read_csv(train_behaviors_path, sep='\t', header=None, names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
        dev_behaviors_df = pd.read_csv(dev_behaviors_path, sep='\t', header=None, names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])

        behaviors_df = pd.concat([train_behaviors_df, dev_behaviors_df])

        behaviors_df['History'] = behaviors_df['History'].fillna('').str.split()
        behaviors_df['Impressions'] = behaviors_df['Impressions'].fillna('').str.split().apply(lambda x: [tuple(imp.split('-')) for imp in x])
        behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'])
        
        return behaviors_df

    def create_user_item_interactions(self):
        user_ids = self.behaviors_data['UserID'].unique()
        news_ids = self.news_data['NewsID'].unique()
        
        user_map = {uid: i for i, uid in enumerate(user_ids)}
        news_map = {nid: i for i, nid in enumerate(news_ids)}
        
        rows, cols, data = [], [], []
        
        for _, row in self.behaviors_data.iterrows():
            user_idx = user_map[row['UserID']]
            for news_id, label in row['Impressions']:
                if news_id in news_map:
                    news_idx = news_map[news_id]
                    rows.append(user_idx)
                    cols.append(news_idx)
                    data.append(int(label))
                    
        return csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(news_ids)))

    def get_article_text(self, news_id):
        article = self.news_data[self.news_data['NewsID'] == news_id]
        if not article.empty:
            return article.iloc[0]['Text']
        return None

    def create_temporal_splits(self):
        train_ratio = self.config['training']['train_ratio']
        val_ratio = self.config['training']['val_ratio']
        
        sorted_behaviors = self.behaviors_data.sort_values(by='Time')
        
        train_end = int(len(sorted_behaviors) * train_ratio)
        val_end = int(len(sorted_behaviors) * (train_ratio + val_ratio))
        
        train_indices = sorted_behaviors.index[:train_end]
        val_indices = sorted_behaviors.index[train_end:val_end]
        test_indices = sorted_behaviors.index[val_end:]
        
        return train_indices, val_indices, test_indices

    def get_user_history(self, user_id, max_history=50):
        history = self.behaviors_data[self.behaviors_data['UserID'] == user_id]['History'].explode().tolist()
        return history[-max_history:]