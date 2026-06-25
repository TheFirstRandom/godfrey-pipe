import warnings
from dotenv import load_dotenv

# warning filters AI generated (see index 1 in docs)
warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout option.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

load_dotenv()

from . import server
from . import models
from . import data
from .main import cli