from pathlib import Path

from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger()


class PreprocessingPipeline:

    def __init__(
        self,
        input_dir="data/raw",
        output_dir="data/processed",
        image_size=(224, 224),
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path):

        image_path = Path(image_path)

        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size)

        output_file = self.output_dir / image_path.name
        image.save(output_file)

        return output_file

    def run(self, input_path=None, output_path=None):

        # Caso o teste envie apenas uma imagem
        if input_path is not None:

            if output_path is not None:
                self.output_dir = Path(output_path)
                self.output_dir.mkdir(parents=True, exist_ok=True)

            return self.process_image(input_path)

        # Funcionamento normal da pipeline
        supported = ("*.jpg", "*.jpeg", "*.png", "*.tif")

        processed = []

        for pattern in supported:
            for image in self.input_dir.glob(pattern):
                processed.append(self.process_image(image))

        return processed