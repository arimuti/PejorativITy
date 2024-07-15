from pathlib import Path

from components.data_loader import PejorativeLoader
from components.pipeline import AnchorFrequencyAnalyzer
from components.processor import FrequencyProcessor

if __name__ == '__main__':
    data_path = Path(__file__).parent.parent.resolve().joinpath('datasets', 'pejorative')
    anchor_path = data_path.joinpath('anchors.json')

    data_loader = PejorativeLoader(data_path=data_path, filename='pejorativity.csv')
    processor = FrequencyProcessor()

    analyzer = AnchorFrequencyAnalyzer(data_loader=data_loader,
                                       processor=processor,
                                       anchor_path=anchor_path)
    analyzer.run()
