from pathlib import Path

from code.data_loader import PejorativeLoader
from code.pipeline import EmbeddingAnalyzer
from code.processor import TransformerProcessor, DataProcessor, AnchorProcessor

if __name__ == '__main__':
    preloaded_model_name = 'm-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0'
    data_loader_args = {
        'batch_size': 4,
        'num_workers': 0
    }

    data_path = Path(__file__).parent.parent.resolve().joinpath('datasets', 'pejorative')
    anchor_path = data_path.joinpath('anchors.json')

    data_loader = PejorativeLoader(data_path=data_path, filename='pejorativity.csv')

    transformer_processor = TransformerProcessor(preloaded_model_name=preloaded_model_name)
    data_processor = DataProcessor(data_loader_args=data_loader_args,
                                   pad_token_id=transformer_processor.tokenizer.pad_token_id)
    anchor_processor = AnchorProcessor(data_loader_args=data_loader_args,
                                       pad_token_id=transformer_processor.tokenizer.pad_token_id)

    analyzer = EmbeddingAnalyzer(
        preloaded_model_name=preloaded_model_name,
        filename='analysis_embedding.pkl',
        data_loader=data_loader,
        transformer_processor=transformer_processor,
        data_processor=data_processor,
        anchor_path=anchor_path,
        anchor_processor=anchor_processor,
        umap_args={'n_components': 2,
                   'min_dist': 0.,
                   'n_neighbors': 50,
                   'metric': 'cosine'
                   },
        similarity_metric='cosine_similarity',
        do_visualization=True,
        do_metrics=False,
        do_similarity=True
    )
    analyzer.run()
