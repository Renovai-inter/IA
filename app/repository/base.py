from abc import ABC, abstractmethod
from uuid import UUID

class Repository[T](ABC):
    def __init__(self, perfil_id: UUID, entity: T):
        self.perfil_id = perfil_id
        self.entity = entity

    @abstractmethod
    def get_snapshot(self):
        raise NotImplementedError
