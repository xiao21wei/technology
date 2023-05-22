import pandas as pd


class Prophet(ABC):
    def __init__(self, name=None):
        self.name = name
        self.data: pd.DataFrame = None
        self.params = {}
