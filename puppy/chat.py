import os
import openai
import requests
from abc import ABC, abstractmethod
from typing import Callable, Union, List, Dict, Any

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatBase(ABC):
    """
    An abstract base class for chat endpoints, typically used
    to interface with an LLM. Subclasses must implement the __call__ method
    (synchronous text generation) and guardrail_endpoint method (for usage
    with guardrails-based validation).
    """

    @abstractmethod
    def __call__(self, text: str, **kwargs) -> str:
        """Synchronous text generation."""
        pass

    @abstractmethod
    def guardrail_endpoint(self) -> Callable[[str], str]:
        """Returns a function that is suitable for usage in guardrails calls."""
        pass


class ChatTogetherEndpoint(ChatBase):
    """
    Uses the Together API to generate text from a model such as LLaMA-2.
    """

    def __init__(
        self,
        api_key: Union[str, None] = None,
        model: str = "togethercomputer/llama-2-70b-chat",
        max_tokens: int = 1000,
        stop: Union[List[str], None] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        request_timeout: int = 600,
    ) -> None:
        """
        Initialize a Together API-based chat endpoint.

        Args:
            api_key (str, optional): Together API key. If None, attempts
                to use TOGETHER_API_KEY from environment.
            model (str): Name of the model at Together to use.
            max_tokens (int): Maximum tokens for the response.
            stop (List[str], optional): Stop sequences. Defaults to ["<human>"].
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling proportion.
            top_k (int): Top-k sampling cutoff.
            repetition_penalty (float): Penalty for repeated tokens.
            request_timeout (int): Timeout for requests in seconds.
        """
        self.stop = stop or ["<human>"]
        self.api_key = os.getenv("TOGETHER_API_KEY") if api_key is None else api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.request_timeout = request_timeout

        self.end_point = "https://api.together.xyz/inference"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def parse_response(self, response: requests.Response) -> str:
        """Parse the text from the JSON response."""
        return response.json()["output"]["choices"][0]["text"]

    def __call__(self, text: str, **kwargs) -> str:
        """
        Generate text given an input prompt.

        Args:
            text (str): The user input text.

        Returns:
            str: The model's response text.
        """
        transaction_payload = {
            "model": self.model,
            "prompt": f"<human>: {text}\n<bot>:",
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

        response = requests.post(
            self.end_point,
            json=transaction_payload,
            headers=self.headers,
            timeout=self.request_timeout,
        )
        response.raise_for_status()

        return self.parse_response(response)

    def guardrail_endpoint(self) -> Callable[[str], str]:
        """
        Return a callable that can be used by guardrails to produce text.

        Returns:
            Callable[[str], str]: A function that calls this instance with an input prompt.
        """
        return self


class ChatOpenAIEndpoint(ChatBase):
    """
    A chat endpoint that uses OpenAI's ChatCompletion API (e.g. GPT-4).
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.0,
    ):
        """
        Args:
            model_name (str): Which OpenAI chat model to use (e.g. "gpt-3.5-turbo" or "gpt-4").
            temperature (float): Sampling temperature for output variability.
        """
        self.model_name = model_name
        self.temperature = temperature

    def __call__(self, text: str, **kwargs) -> str:
        """
        Direct synchronous call for text generation, if needed outside guardrails usage.

        Args:
            text (str): The user prompt.

        Returns:
            str: The generated text.
        """
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]["content"]

    def guardrail_endpoint(self) -> Callable[[str], str]:
        """
        Return a callable that guardrails can use, injecting a system prompt if needed.

        Returns:
            Callable[[str], str]: Function that calls the ChatCompletion endpoint.
        """

        def endpoint_func(user_input: str, **kwargs) -> str:
            return self.__call__(user_input)

        return endpoint_func


def get_chat_end_points(
    end_point_type: str,
    chat_config: Dict[str, Any]
) -> Union[ChatOpenAIEndpoint, ChatTogetherEndpoint]:
    """
    Factory function that returns a chat endpoint (OpenAI or Together).

    Args:
        end_point_type (str): Either "openai" or "together".
        chat_config (Dict[str, Any]): Configuration dict with relevant parameters.

    Returns:
        Union[ChatOpenAIEndpoint, ChatTogetherEndpoint]: The configured chat endpoint object.

    Raises:
        NotImplementedError: If an unsupported endpoint type is provided.
    """
    match end_point_type:
        case "openai":
            return ChatOpenAIEndpoint(
                model_name=chat_config["model_name"],
                temperature=chat_config["temperature"],
            )
        case "together":
            return ChatTogetherEndpoint(
                api_key=chat_config.get("api_key"),
                model=chat_config["model"],
                max_tokens=chat_config["max_tokens"],
                stop=chat_config.get("stop"),
                temperature=chat_config["temperature"],
                top_p=chat_config["top_p"],
                top_k=chat_config["top_k"],
                repetition_penalty=chat_config["repetition_penalty"],
                request_timeout=chat_config["request_timeout"],
            )
        case _:
            raise NotImplementedError(
                f"Chat endpoint type '{end_point_type}' is not implemented."
            )
