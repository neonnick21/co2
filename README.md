## Setup
1. Install dependencies:  
   `pip install -r requirements.txt`  
2. Download BCCD dataset:  
   `[robodata download bccd-coco --split](https://public.roboflow.com/object-detection/bccd/4/download/coco)`  
3. Train model:  
   `python src/train.py --data_dir data/ --epochs 100`  
4. Evaluate:  
   `python src/eval.py --weights saved_models/rf_detr_best.pth`  
