import os
import asyncio
import numpy as np
from typing import List, Union
from langchain.embeddings import OpenAIEmbeddings


class OpenAILongerThanContextEmb:
    """
    An embedding class that uses OpenAI as the embedding backend. If the input
    text exceeds the model's context size, the input is split into smaller
    chunks of size `chunk_size` and each chunk is embedded separately. The
    final embedding is the average of these chunk embeddings.

    Reference:
        https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    """

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 5000,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the OpenAILongerThanContextEmb object.

        Args:
            openai_api_key (str, optional): The API key for OpenAI. If not provided,
                the key is retrieved from the environment variable OPENAI_API_KEY.
            embedding_model (str, optional): The OpenAI model to use for embedding.
                Defaults to "text-embedding-ada-002".
            chunk_size (int, optional): The maximum number of tokens to send
                to the embedding model per request. Defaults to 5000.
            verbose (bool, optional): Whether to show a progress bar in LangChain.
                Defaults to False.
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.emb_model = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.openai_api_key,
            chunk_size=chunk_size,
            show_progress_bar=verbose,
        )

    async def _embed_async(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Asynchronously embed one or more text strings.

        Args:
            text (List[str] or str): A list of text or a single string.

        Returns:
            List[List[float]]: A list of embeddings (one for each input).
        """
        if isinstance(text, str):
            text = [text]
        return await self.emb_model.aembed_documents(texts=text, chunk_size=None)

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        Embed a list of text or a single text string synchronously.

        Args:
            text (List[str] or str): The text(s) to embed.

        Returns:
            np.ndarray: A NumPy array of embeddings (shape: [num_texts, embedding_dimension]).
        """
        return np.array(asyncio.run(self._embed_async(text))).astype("float32")

    def get_embedding_dimension(self) -> int:
        """
        Return the dimension of the embedding for the specified model.

        Raises:
            NotImplementedError: If the embedding dimension for the given model is not implemented.

        Returns:
            int: Dimension size of the embedding vector.
        """
        match self.emb_model.model:
            case "text-embedding-ada-002":
                return 1536
            case _:
                raise NotImplementedError(
                    f"Embedding dimension for model {self.emb_model.model} is not implemented."
                )
