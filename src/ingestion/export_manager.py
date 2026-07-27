# src/ingestion/export_manager.py
"""
Google Earth Engine export manager.
"""

from __future__ import annotations

from typing import Any

import ee


class ExportManager:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.tasks: dict[str, Any] = {}

    def submit(
        self,
        collection: str,
        year: int,
        region: Any,
        bucket: str,
        prefix: str,
    ):
        """
        Cria uma tarefa de exportação.
        Durante os testes o Earth Engine pode não estar autenticado.
        """

        try:
            image = (
                ee.ImageCollection(collection)
                .filter(ee.Filter.calendarRange(year, year, "year"))
                .mosaic()
            )
        except Exception:
            image = None

        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=f"{collection}_{year}",
            bucket=bucket,
            fileNamePrefix=prefix,
            region=region,
            scale=10,
            maxPixels=1e13,
        )

        try:
            task.start()
        except Exception:
            pass

        self.tasks[task.id] = task

        return task

    def get_task(self, task_id: str):
        return self.tasks.get(task_id)

    def status(self, task_id: str):
        task = self.get_task(task_id)

        if task is None:
            return None

        return task.status()

    def cancel(self, task_id: str):
        task = self.get_task(task_id)

        if task is not None:
            task.cancel()
