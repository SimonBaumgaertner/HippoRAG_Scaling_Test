import json
import os
import zipfile

def process_dataset():
    zip_path = "test_data/HotpotQA_Dev.zip"
    output_dir = "HotpotQA_Dev"
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return

    print(f"Extracting {zip_path} to {output_dir}...")
    # Clean output dir if needed? user might have files there. safest is to overwrite/add.
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Extraction complete.")
    
    # Process corpus
    corpus_list = []
    
    # Walk through the directory to find document folders
    print("Processing documents...")
    items = os.listdir(output_dir)
    processed_count = 0
    
    for item in items:
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            doc_id = item
            # The raw file name typically matches the folder name
            raw_file = os.path.join(item_path, f"{doc_id}_raw.txt")
            
            # Fallback if filename is slightly different or to be robust
            if not os.path.exists(raw_file):
                # Try finding any _raw.txt
                files = os.listdir(item_path)
                raw_files = [f for f in files if f.endswith('_raw.txt')]
                if raw_files:
                    raw_file = os.path.join(item_path, raw_files[0])
            
            if os.path.exists(raw_file):
                try:
                    with open(raw_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    corpus_list.append({"title": doc_id, "text": text})
                    processed_count += 1
                except Exception as e:
                    print(f"Error reading {raw_file}: {e}")
            
    print(f"Extracted {processed_count} documents.")
    
    # Process queries
    qa_file = os.path.join(output_dir, "QA.json")
    queries = []
    if os.path.exists(qa_file):
        print(f"Processing queries from {qa_file}...")
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                raw_queries = json.load(f)
                
            for q in raw_queries:
                queries.append({
                    "id": q.get("question_id"),
                    "question": q.get("question"),
                    "answer": q.get("correct_answer"),
                })
            print(f"Processed {len(queries)} queries.")
        except Exception as e:
            print(f"Error reading {qa_file}: {e}")
    else:
        print(f"Warning: {qa_file} not found.")

    # Save corpus
    corpus_path = os.path.join(output_dir, "hotpotqa_corpus.json")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_list, f, indent=2)
    print(f"Saved corpus to {corpus_path}")
    
    # Save queries
    queries_path = os.path.join(output_dir, "hotpotqa.json")
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2)
    print(f"Saved queries to {queries_path}")

if __name__ == "__main__":
    process_dataset()
