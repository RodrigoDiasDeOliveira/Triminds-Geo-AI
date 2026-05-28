from pathlib import Path
from PIL import Image

from src.utils.logger import setup_logger


logger = setup_logger()


class PreprocessingPipeline:

    def __init__(
        self,
        input_dir,
        output_dir,
        image_size=(224, 224)
    ):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def process_image(self, image_path):

        try:

            image = Image.open(image_path).convert("RGB")

            image = image.resize(self.image_size)

            output_path = self.output_dir / image_path.name

            image.save(output_path)

            logger.info(f"Processed: {image_path.name}")

        except Exception as error:

            logger.error(
                f"Error processing {image_path.name}: {error}"
            )

    def run(self):

        supported_formats = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.tif"
        ]

        for extension in supported_formats:

            for image_path in self.input_dir.glob(extension):

                self.process_image(image_path)


if __name__ == "__main__":

    pipeline = PreprocessingPipeline(
        input_dir="data/raw",
        output_dir="data/processed"
    )

    pipeline.run()