from dataclasses import dataclass

from topobenchmarkx.data.basedata import BaseData


@dataclass
class HypergraphData(BaseData):
    name: str
    unit_price: float
    quantity_on_hand: int = 0
