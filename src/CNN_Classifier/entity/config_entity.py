#update the entity
from dataclasses import dataclass
from pathlib import Path

#dataclass
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path