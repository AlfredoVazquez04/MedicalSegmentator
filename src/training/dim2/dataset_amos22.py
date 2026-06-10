import lightning as L

class DatasetAmos22(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()