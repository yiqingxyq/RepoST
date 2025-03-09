from abc import ABC, abstractmethod
from warnings import warn

from datasets import load_dataset


class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # DATASET_PATH: str = None

    # # The name of a subset within `DATASET_PATH`.
    # DATASET_NAME: str = None

    def __init__(
        self, 
        dataset_path: str = None, dataset_name: str = None, data_files: dict = None, 
        stop_words=None, requires_execution=True, cache_dir=None,
    ):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            optional_args = {"cache_dir": cache_dir}
            if dataset_name is not None:
                optional_args["name"] = dataset_name
            if data_files is not None:
                optional_args["data_files"] = data_files
            print("Loading dataset ..")
            self.dataset = load_dataset(dataset_path, **optional_args)
        except Exception as e:
            warn(
                f"Loading the dataset failed with {str(e)}. This task will use a locally downloaded dataset, not from the HF hub. \
                This is expected behavior for the DS-1000 benchmark but not for other benchmarks!"
            )

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []
    
    def preprocess_all_data(self, **kwargs):
        """Preprocess all data (for vllm inference)"""
        pass

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx, new_tokens_only=False):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]
