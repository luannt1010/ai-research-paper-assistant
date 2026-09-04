from __future__ import annotations

from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage

class BaseLLM(ABC):
    def __init__(self, model_name: str, temperature: float, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs

        self.model: BaseChatModel | None = None

    @abstractmethod
    def _build(self) -> BaseChatModel:
        pass

    @property
    def generator(self) -> BaseChatModel:
        if self.model is None:
            self.model = self._build()
        return self.model

    def generate(self, message: LanguageModelInput) -> BaseMessage:
        return self.generator.invoke(message)