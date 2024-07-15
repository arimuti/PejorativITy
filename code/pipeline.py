from itertools import chain
from pathlib import Path
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification
from umap import UMAP

from utility import load_json, save_json, save_pickle
from components.data_loader import DataLoader
from components.processor import TransformerProcessor, DataProcessor, AnchorProcessor, FrequencyProcessor


class EmbeddingAnalyzer:

    def __init__(
            self,
            preloaded_model_name: Union[Path, str],
            data_loader: DataLoader,
            transformer_processor: TransformerProcessor,
            data_processor: DataProcessor,
            anchor_path: Path,
            anchor_processor: AnchorProcessor,
            umap_args,
            filename: str = 'analysis_embeddings.pkl',
            similarity_metric: str = 'cosine_similarity',
            do_visualization: bool = True,
            do_similarity: bool = True,
            do_metrics: bool = False
    ):
        self.preloaded_model_name = preloaded_model_name
        self.data_loader = data_loader
        self.transformer_processor = transformer_processor
        self.data_processor = data_processor
        self.anchor_path = anchor_path
        self.anchor_processor = anchor_processor
        self.umap_args = umap_args
        self.filename = filename
        self.similarity_metric = similarity_metric
        self.do_visualization = do_visualization
        self.do_similarity = do_similarity
        self.do_metrics = do_metrics

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        self.bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=self.preloaded_model_name)
        self.bert_config.output_hidden_states = True
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.preloaded_model_name,
            config=self.bert_config).to(device)

        self.similarity_map = {
            'cosine_similarity': self.cosine_similarity
        }

    def cosine_similarity(
            self,
            data_embeddings,
            anchor_embeddings,
    ):
        if len(data_embeddings.shape) == 1:
            data_embeddings = data_embeddings.reshape(1, -1)
        if len(anchor_embeddings.shape) == 1:
            anchor_embeddings = anchor_embeddings.reshape(1, -1)

        return cosine_similarity(data_embeddings, anchor_embeddings)

    def group_visualize(
            self,
            data_embeddings,
            data_labels,
            data_predictions,
            data_markers,
            anchor_embeddings=None,
            anchors=None,
            anchor_map=None,
            serialize: bool = False,
            serialization_path: Path = None
    ):
        # data_embeddings   ->    [#samples, dim]
        # anchor_embeddings ->    [#anchors, dim]

        data_embeddings = np.array(data_embeddings) if type(data_embeddings) != np.ndarray else data_embeddings
        anchor_embeddings = np.array(anchor_embeddings) if type(anchor_embeddings) != np.ndarray else anchor_embeddings

        # Fit all embeddings
        mapper = UMAP(**self.umap_args)
        mapper_fit_data = data_embeddings if anchor_embeddings is None else np.concatenate((data_embeddings,
                                                                                            anchor_embeddings), axis=0)
        mapper.fit(X=mapper_fit_data)

        enc_data_embeddings = mapper.transform(data_embeddings)
        axis_x_limit = (np.min(enc_data_embeddings[:, 0]), np.max(enc_data_embeddings[:, 0]))
        axis_y_limit = (np.min(enc_data_embeddings[:, 1]), np.max(enc_data_embeddings[:, 1]))

        data_labels = np.array(data_labels)
        data_predictions = np.array(data_predictions)

        fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
        ax.set_title(f'Word analysis', fontsize=24)
        ax.set_xlim(left=axis_x_limit[0] - 1.0, right=axis_x_limit[1] + 1.0)
        ax.set_ylim(bottom=axis_y_limit[0] - 1.0, top=axis_y_limit[1] + 1.0)

        # Plot individual groups
        legend_handles = []
        legend_labels = []
        for search_word in set(data_labels):
            if search_word not in anchor_map:
                print(f'Skipping visualization for search word {search_word} since no anchors were found.')
                continue

            search_marker = data_markers[search_word]

            search_indexes = np.where(data_labels == search_word)[0]
            search_embeddings = enc_data_embeddings[search_indexes]
            search_predictions = data_predictions[search_indexes]

            ax.scatter(x=search_embeddings[:, 0],
                       y=search_embeddings[:, 1],
                       c=['r' if pred == 0 else 'g' for pred in search_predictions],
                       marker=search_marker,
                       label=f'{search_word}')
            legend_handles.append(plt.plot([], [],
                                           marker=search_marker,
                                           linestyle='None',
                                           markerfacecolor='w',
                                           markeredgecolor='k')[0])
            legend_labels.append(search_word)

        legend_handles.extend([mpatches.Patch(color='r'),
                               mpatches.Patch(color='g')])
        legend_labels.extend(['!pejorative', 'pejorative'])
        ax.legend(handles=legend_handles,
                  labels=legend_labels)

        if anchor_embeddings is not None:
            anchor_embeddings = mapper.transform(anchor_embeddings)

            ax.scatter(x=anchor_embeddings[:, 0],
                       y=anchor_embeddings[:, 1],
                       c='black',
                       marker='x')

            # Annotate anchors
            for idx, anchor_text in enumerate(anchors):
                anchor_emb = anchor_embeddings[idx]
                ax.annotate(anchor_text, xy=(anchor_emb[0], anchor_emb[1]))

        if serialize and serialization_path is not None:
            plt.savefig(serialization_path.joinpath(f'word_analysis.png'),
                        bbox_inches='tight',
                        dpi=100)

        plt.show()
        plt.close(fig)

    def group_distances(
            self,
            data_embeddings,
            data_labels,
            data_predictions,
            anchor_embeddings=None,
            anchors=None,
            anchor_map=None
    ):
        # data_embeddings   ->    [#samples, dim]
        # anchor_embeddings ->    [#anchors, dim]

        data_embeddings = np.array(data_embeddings) if type(data_embeddings) != np.ndarray else data_embeddings
        anchor_embeddings = np.array(anchor_embeddings) if type(anchor_embeddings) != np.ndarray else anchor_embeddings

        data_labels = np.array(data_labels)
        data_predictions = np.array(data_predictions)

        similarity_info = {}

        # Plot individual groups
        for search_word in sorted(list(set(data_labels))):
            if search_word not in anchor_map:
                print(f'Skipping computation for search word {search_word} since no anchors were found.')
                continue

            search_indexes = np.where(data_labels == search_word)[0]
            search_embeddings = data_embeddings[search_indexes]
            search_predictions = data_predictions[search_indexes]

            word_anchors = sorted(anchor_map[search_word])
            anchor_indexes = [anchors.index(anchor) for anchor in word_anchors]
            search_anchors = anchor_embeddings[anchor_indexes]

            # Positive
            positive_embeddings = search_embeddings[np.where(search_predictions == 1)[0]]
            positive_mean, positive_std = 0.0, 0.0
            if len(positive_embeddings):
                positive_similarity = self.similarity_map[self.similarity_metric](positive_embeddings, search_anchors)
                positive_mean, positive_std = np.mean(positive_similarity, axis=0), np.std(positive_similarity, axis=0)
                positive_mean, positive_std = np.nan_to_num(positive_mean), np.nan_to_num(positive_std)

            negative_embeddings = search_embeddings[np.where(search_predictions == 0)[0]]
            negative_mean, negative_std = 0.0, 0.0
            if len(negative_embeddings):
                negative_similarity = self.similarity_map[self.similarity_metric](negative_embeddings, search_anchors)
                negative_mean, negative_std = np.mean(negative_similarity, axis=0), np.std(negative_similarity, axis=0)
                negative_mean, negative_std = np.nan_to_num(negative_mean), np.nan_to_num(negative_std)

            similarity_info[search_word] = {
                'anchors': word_anchors,
                'positive_predictions': len(positive_embeddings),
                'positive_similarity': {
                    anchor: f'{positive_mean[anchor_idx]:.2f} +/- {positive_std[anchor_idx]:.2f}'
                    for anchor_idx, anchor in enumerate(word_anchors)
                } if len(positive_embeddings) else 'N/A',
                'avg_positive_similarity': f'{np.mean(positive_mean):.2f} +/- {np.mean(positive_std):.2f}',
                'negative_predictions': len(negative_embeddings),
                'negative_similarity': {
                    anchor: f'{negative_mean[anchor_idx]:.2f} +/- {negative_std[anchor_idx]:.2f}'
                    for anchor_idx, anchor in enumerate(word_anchors)
                } if len(negative_embeddings) else 'N/A',
                'avg_negative_similarity': f'{np.mean(negative_mean):.2f} +/- {np.mean(negative_std):.2f}',
            }

        return similarity_info

    def extract_data_embeddings(
            self,
            text,
            search_word,
            search_positions,
            data,
            steps
    ):
        data_embeddings = {}

        self.bert.eval()

        with tqdm(range(steps), leave=True, position=0, desc='Extracting search word embeddings') as pbar:
            for batch_x, batch_y in data._get_iterator():
                batch_y = batch_y.detach().cpu().numpy()

                model_output = self.bert(input_ids=batch_x['text_ids'],
                                         attention_mask=batch_x['text_mask'])

                # [bs,]
                batch_predictions = th.argmax(model_output.logits, dim=-1).detach().cpu().numpy()

                # [bs, # tokens, dim]
                batch_embeddings = model_output.hidden_states[-1].detach().cpu().numpy()

                batch_ids = batch_x['sample_id'].detach().cpu().numpy()
                batch_positions = search_positions[batch_ids[0]:batch_ids[-1] + 1]

                for sample_embedding, sample_positions, sample_id, sample_pred, sample_truth in zip(batch_embeddings,
                                                                                                    batch_positions,
                                                                                                    batch_ids,
                                                                                                    batch_predictions,
                                                                                                    batch_y):
                    for in_sample_position in sample_positions:
                        search_embedding = np.mean(sample_embedding[in_sample_position], axis=0)

                        data_embeddings.setdefault('Sample id', []).append(sample_id)
                        data_embeddings.setdefault('Text', []).append(text[sample_id])
                        data_embeddings.setdefault('Search word', []).append(search_word[sample_id])
                        data_embeddings.setdefault('Search embedding', []).append(search_embedding)
                        data_embeddings.setdefault('Sample prediction', []).append(sample_pred)
                        data_embeddings.setdefault('Sample truth', []).append(sample_truth)

                pbar.update(1)

        return data_embeddings

    def extract_anchor_embeddings(
            self,
            data,
            steps
    ):
        anchor_embeddings = None

        self.bert.eval()

        with tqdm(range(steps), leave=True, position=0, desc='Extracting anchor embeddings') as pbar:
            for batch in data._get_iterator():

                text_ids = batch['text_ids']
                text_mask = batch['text_mask']

                model_output = self.bert(input_ids=text_ids,
                                         attention_mask=text_mask)

                # [bs, dim]
                batch_embeddings = th.sum(model_output.hidden_states[-1] * text_mask[:, :, None], dim=1) / th.sum(
                    text_mask, dim=1)[:, None]
                batch_embeddings = batch_embeddings.detach().cpu().numpy()

                if anchor_embeddings is None:
                    anchor_embeddings = batch_embeddings
                else:
                    anchor_embeddings = np.concatenate((anchor_embeddings, batch_embeddings), axis=0)

                pbar.update(1)

        return anchor_embeddings

    def run(
            self,
            serialization_path: Path = None,
            serialize: bool = False
    ):
        # Load data
        data = self.data_loader.load_data()

        # Parse data
        processed_data = self.transformer_processor.process(texts=data.text.values,
                                                            search_words=data.search_word.values)
        th_data_loader, th_steps = self.data_processor.process(text_ids=processed_data['text_ids'],
                                                               sample_id=data['sample_id'],
                                                               text_mask=processed_data['text_mask'],
                                                               pejorative=data['pejorative'])

        # Extract word embeddings from data
        data_info = self.extract_data_embeddings(data=th_data_loader,
                                                 steps=th_steps,
                                                 search_word=processed_data['search_word'],
                                                 text=processed_data['text'],
                                                 search_positions=processed_data['search_positions'])

        # Extract anchor embeddings (if any)
        if self.anchor_path.exists() and self.anchor_path.is_file():
            anchor_map = load_json(self.anchor_path)
            anchor_markers = load_json(self.anchor_path.with_name('anchor_markers.json'))
            anchors = list(chain(*list(anchor_map.values())))
            anchors = list(set(anchors))
            anchor_tokens = self.transformer_processor.tokenizer(anchors)
            anchor_th_data_loader, anchor_th_steps = self.anchor_processor.process(text_ids=anchor_tokens['input_ids'],
                                                                                   text_mask=anchor_tokens['attention_mask'])
            anchor_embeddings = self.extract_anchor_embeddings(data=anchor_th_data_loader,
                                                               steps=anchor_th_steps)
        else:
            anchor_map = None
            anchor_markers = None
            anchors = None
            anchor_embeddings = None

        if serialize and serialization_path is not None:
            save_pickle(serialization_path.joinpath('data_embeddings.pkl'), data_info)
            save_pickle(serialization_path.joinpath('anchor_embeddings.pkl'), {
                'anchors': anchors,
                'anchor_map': anchor_map,
                'embeddings': anchor_embeddings
            })

        # Compute metrics
        if self.do_metrics:
            f1 = f1_score(y_true=np.array(data_info['Sample truth']),
                          y_pred=np.array(data_info['Sample prediction']),
                          average='binary',
                          pos_label=1)
            print(f'F1-score: {f1}')

            if serialize and serialization_path is not None:
                save_json(serialization_path.joinpath('metrics.json'), {
                    'f1': f1
                })

        # Visualize embeddings
        if self.do_visualization:
            self.group_visualize(data_embeddings=data_info['Search embedding'],
                                 data_labels=data_info['Search word'],
                                 data_predictions=data_info['Sample truth'],
                                 data_markers=anchor_markers,
                                 anchors=anchors,
                                 anchor_embeddings=anchor_embeddings,
                                 anchor_map=anchor_map,
                                 serialize=serialize,
                                 serialization_path=serialization_path)

        # Compute distances
        if self.do_similarity:
            similarity_info = self.group_distances(data_embeddings=data_info['Search embedding'],
                                                   data_labels=data_info['Search word'],
                                                   data_predictions=data_info['Sample truth'],
                                                   anchors=anchors,
                                                   anchor_embeddings=anchor_embeddings,
                                                   anchor_map=anchor_map)
            print(similarity_info)

            if serialize and serialization_path is not None:
                save_json(serialization_path.joinpath('similarity_info.json'), similarity_info)


class AnchorFrequencyAnalyzer:

    def __init__(
            self,
            data_loader: DataLoader,
            processor: FrequencyProcessor,
            anchor_path: Path
    ):
        self.data_loader = data_loader
        self.processor = processor
        self.anchor_path = anchor_path

    def run(
            self,
            serialization_path: Path = None,
            serialize: bool = False
    ):
        # Load data
        data = self.data_loader.load_data()

        # Load anchors
        anchor_map = load_json(self.anchor_path)
        anchors = list(chain(*list(anchor_map.values())))

        # Processor
        frequency_info = self.processor.process(texts=data.text.values,
                                                anchors=anchors,
                                                anchor_map=anchor_map,
                                                search_words=data.search_word.values)

        print(frequency_info)

        if serialize and serialization_path is not None:
            save_json(serialization_path.joinpath('frequency_info.json'), frequency_info)
