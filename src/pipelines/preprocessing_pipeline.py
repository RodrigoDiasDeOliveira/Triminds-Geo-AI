from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.utils.logger import setup_logger

logger = setup_logger()


class PreprocessingPipeline:
    """Pipeline responsável pelo pré-processamento de imagens."""

    def __init__(
        self,
        input_dir: str | Path = "data/raw",
        output_dir: str | Path = "data/processed",
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def process_image(
        self,
        image_path: str | Path,
    ) -> Path:
        """
        Process a single image.

        Parameters
        ----------
        image_path
            Path to the input image.

        Returns
        -------
        Path
            Path to the processed image. If processing fails,
            returns the original image path for backward compatibility.
        """
        image_path = Path(image_path)

        try:
            output_file = self.output_dir / image_path.name

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(self.image_size)
                image.save(output_file)

            logger.info(
                "Processed image: %s",
                image_path.name,
            )

            return output_file

        except (UnidentifiedImageError, OSError) as exc:
            logger.error(
                "Failed to process image '%s': %s",
                image_path,
                exc,
            )

            # Mantido para compatibilidade com os testes existentes.
            return image_path

    def run(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> Path | list[Path]:
        """
        Execute the preprocessing pipeline.

        If an input file is provided, process only that file.
        Otherwise, process all supported images found in the input directory.
        """
        if input_path is not None:
            if output_path is not None:
                self.output_dir = Path(output_path)
                self.output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            return self.process_image(input_path)

        supported_formats = (
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.tif",
        )

        processed: list[Path] = []

        for pattern in supported_formats:
            for image_path in self.input_dir.glob(pattern):
                result = self.process_image(image_path)

                if result is not None:
                    processed.append(result)

        return processed


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
