# Breast Cancer Histopathological Image Classification

This repository contains code and notebooks for classifying breast cancer images using deep learning models. The goal is to develop a reliable model to assist in early diagnosis based on histopathology images.

---

#### Dataset

The project uses the [Breast Cancer Histopathological Database (BreakHis)](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/).
> **Note:** The BreaKHis dataset is large and must be downloaded manually or via the Kaggle API. 

  **Option 1:** Manual download
  - Download the dataset from Kaggle: [BreaKHis Dataset](https://www.kaggle.com/ambarish/breakhis)
  - Extract it, and place the extracted folders under the `data/` directory, keeping the structure:
       -      data/
              ├── benign/
              ├── malignant/
              └── ...
    
  **Option 2:** Download using Kaggle API (example in Colab)
  - You can use the Kaggle API in your notebook to download and prepare the dataset:
      ```python
        # Upload kaggle.json file (your API token)
        from google.colab import files
        import os, shutil, glob
        
        files.upload()  # Upload kaggle.json
        
        os.makedirs("/root/.kaggle", exist_ok=True)
        shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")
        os.chmod("/root/.kaggle/kaggle.json", 600)
        
        # Download BreaKHis dataset using kagglehub
        import kagglehub
        path = kagglehub.dataset_download("ambarish/breakhis")
        
        # Copy dataset to working directory
        dataset_images_path = os.path.join(path, "BreaKHis_v1", "BreaKHis_v1", "histology_slides", "breast")
        destination = "/content/breakhis_dataset"
        if not os.path.exists(destination):
            shutil.copytree(dataset_images_path, destination)
        
        print("Dataset ready at:", destination)
        ``` 


#### Project Structure
- **`src/`**: Python scripts for loading, preprocessing, training, evaluation, and utilities.  
- **`models/`**: Saved model checkpoints (can be empty initially).  
- **`notebooks/`**: Jupyter notebooks for experiments and analysis.

#### Prerequisites

Make sure you have Python 3.7+ installed.

#### Install dependencies

```bash
pip install -r requirements.txt
```

