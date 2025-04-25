import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Union

from typing_extensions import override

if TYPE_CHECKING:
    import mlflow


class MlflowHandler(logging.Handler):
    def __init__(
        self, client: "mlflow.tracking.MlflowClient", run_id: str, level: Union[int, str] = 0
    ) -> None:
        super().__init__(level)
        self._tempfile = None
        self._file_handler = None
        self._client = client
        self._run_id = run_id

    @property
    def file_handler(self) -> logging.FileHandler:
        if self._tempfile is None:
            self._tempfile = tempfile.mkdtemp()
            self._file_handler = logging.FileHandler(os.path.join(self._tempfile, "reax.log"))

        return self._file_handler

    @override
    def emit(self, record) -> None:
        self.file_handler.emit(record)
        self._client.log_artifact(self._run_id, self._file_handler.baseFilename)

    def close(self):
        if self._tempfile is not None:
            self._file_handler.close()
            shutil.rmtree(self._tempfile)
        self._client = None
        self._run_id = None
