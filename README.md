## Setup
1. Install dependencies:  
   `pip install -r requirements.txt`  
2. Download BCCD dataset:  
   `curl -L "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip`  
3. Train model:  
   `python src/train.py --data_dir data/ --epochs 100`  
4. Evaluate:  
   `python src/eval.py --weights saved_models/rf_detr_best.pth`  
