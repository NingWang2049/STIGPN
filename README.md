# STIGPN
Space-Time Interaction Graph Parsing Networks for Human-Object Interaction Recognition，ACM MM'21
### Installation
1. Clone this repository.   
    ```
    git clone https://github.com/GuangmingZhu/STIGPN.git
    ```
  
2. Install Python dependencies:   
    ```
    pip install -r requirements.txt
    ```
### Prepare Data
1. Follow [here](https://github.com/praneeth11009/LIGHTEN-Learning-Interactions-with-Graphs-and-Hierarchical-TEmporal-Networks-for-HOI) to prepare the original data of CAD120 dataset in `CAD120/datasets` folder.
2. You can also download the data we have processed directly from [here]{https://pan.baidu.com/s/1uRP7GCBBL5Eb0Fi7c362oA}.
3. We also provide some checkpoints to the trained models. Download them [here]{https://pan.baidu.com/s/1uRP7GCBBL5Eb0Fi7c362oA} and put them in the checkpoints folder
4. extraction code: 1rly
### Training
For the CAD120 dataset:
    ```
        python train_CAD120.py --model VisualModelV
    ```
    ```
        python train_CAD120.py --model SemanticModelV 
    ```

### Testing
For the CAD120 dataset:
    ```
        python eval_CAD120.py
    ```
