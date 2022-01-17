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
2. You can also download the data we have processed directly from [here](https://drive.google.com/drive/folders/1ntHUZO8CBHV6-Wwci6XqcaQdK4EJUZJi?usp=sharing).
3. We also provide some checkpoints to the trained models. Download them [here](https://drive.google.com/drive/folders/1ntHUZO8CBHV6-Wwci6XqcaQdK4EJUZJi?usp=sharing) and put them in the checkpoints folder
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
# Citation
If you use our annotations in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@inproceedings{wang2021spatio,
  title={Spatio-Temporal Interaction Graph Parsing Networks for Human-Object Interaction Recognition},
  author={Wang, Ning and Zhu, Guangming and Zhang, Liang and Shen, Peiyi and Li, Hongsheng and Hua, Cong},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={4985--4993},
  year={2021}
}
```
