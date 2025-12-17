import sys
from pathlib import Path

# Add src to python path to allow imports
src_path = Path(__file__).parent
sys.path.append(str(src_path))

from data_classes.data_set import DataSet

def main():
    # Define dataset path
    dataset_path = Path("HotpotQA_Dev")
    if not dataset_path.exists():
        # Fallback for running from src or other locations if PWD is not project root
        project_root = Path(__file__).parent.parent
        dataset_path = project_root / "HotpotQA_Dev"
    
    print(f"Loading dataset from: {dataset_path.absolute()}")
    
    try:
        data_set = DataSet(dataset_path)
        print("Dataset loaded successfully!")
        print(f"Number of documents: {len(data_set.documents)}")
        print(f"Number of QA pairs: {len(data_set.qa_pairs)}")
        
        # Validation checks
        if len(data_set.documents) > 0:
            print(f"Sample Document ID: {data_set.documents[0].id}")
            print(f"Sample Document Title: {data_set.documents[0].title}")
        
        if len(data_set.qa_pairs) > 0:
            print(f"Sample Question: {data_set.qa_pairs[0].question}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
