import pandas as pd
from datasets import load_dataset

def load_huggingface_dataset_to_dataframe(dataset_name: str, subset_name: str) -> pd.DataFrame:
    """
    Loads a Hugging Face dataset and converts it to a Pandas DataFrame.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'yixuantt/MultiHopRAG').
        subset_name (str): Name of the subset to load (e.g., 'corpus', 'MultiHopRAG').
    Returns:
        pd.DataFrame: Pandas DataFrame containing the dataset.
    """
    try:
        dataset = load_dataset(dataset_name, subset_name)
        df = pd.DataFrame(dataset['train'])  # Assuming you want the 'train' split
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


dataset_name = "yixuantt/MultiHopRAG"
subset_name = "corpus"
df_multi_hop_rag = load_huggingface_dataset_to_dataframe(dataset_name, subset_name)
print(df_multi_hop_rag)

# save the dataframe to a csv file
df_multi_hop_rag.to_csv("multi_hop_rag_dataset.csv", index=False)