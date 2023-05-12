# BFC-BL: Few-Shot Classification and Segmentation combining Bi-directional Feature Correlation and Boundary constraint

![result](../../../BMVC会议/images/result.png)

The file structure should be as follows:

```
ifsl/
├── fs-cs
│   ├──  common/
│   ├──  data/
│   ├──  model/
│   ├──  main.py
│   └──  README.md
├── environment.yml
├── BFC-BL_Model_Architecture.png
├── LICENSE
├── result.png
└── README.md
```

## Training a model

### Training with segmentation annotaiton

```
python main.py --datapath YOUR_DATASET_DIR \
               --method {panet, pfenet, hsnet, asnet, bfc-bl} \
               --benchmark {pascal} \
               --logpath YOUR_DIR_TO_SAVE_CKPT \
               --way 1 \
               --shot 1 \
               --fold {0, 1, 2, 3} \
               --backbone {resnet50}
```

## Evaluating authors' checkpoints

```
python main.py --datapath YOUR_DATASET_DIR \
               --method {panet, pfenet, hsnet, asnet, bfc-bl} \
               --benchmark {pascal, coco} \
               --logpath {panet, pfenet, hsnet, asnet, bfc-bl} \
               --way {1, 2, 3, 4, 5} \
               --shot {1, 5} \
               --bsz 1 \
               --fold {0, 1, 2, 3} \
               --backbone {resnet50} \
               --eval
```

## Few-shot classification and segmentation results

**Experimental results on Pascal-5i datasets on the FS-CS task.**

<table>
    <tr>
        <td><center>Methods</center></td>
        <td colspan="2"><center>1-way 1-shot</center></td>
        <td colspan="2"><center>2-way 1-shot</center></td>
    </tr>
    <tr>
        <td ><center>metric</center></td>
        <td ><center>cls. 0/1 ER</center></td>
        <td ><center>seg. mIoU</center></td>
        <td ><center>cls. 0/1 ER</center></td>
        <td ><center>seg. mIoU</center></td>
    </tr>
    <tr>
        <td ><center>PANet</center></td>
        <td ><center>69.0</center></td>
        <td ><center>36.2</center></td>
        <td ><center>50.9</center></td>
        <td ><center>37.2</center></td>
    </tr>
    <tr>
        <td ><center>PFENet</center></td>
        <td ><center>74.6</center></td>
        <td ><center>43.0</center></td>
        <td ><center>41.0</center></td>
        <td ><center>35.3</center></td>
    </tr>
    <tr>
        <td ><center>HSNet</center></td>
        <td ><center>83.7</center></td>
        <td ><center>49.7</center></td>
        <td ><center>67.3</center></td>
        <td ><center>43.5</center></td>
    </tr>
    <tr>
        <td ><center>ASNet</center></center></td>
        <td ><center>84.9</center></td>
        <td ><center>52.3</center></td>
        <td ><center>68.3</center></td>
        <td ><center>47.8</center></td>
    </tr>
    <tr>
        <td ><center>BFC-BL</center></td>
        <td ><center>86.6</center></td>
        <td ><center>53.6</center></td>
        <td ><center>70.1</center></td>
        <td ><center>49.2</center></td>
    </tr>
</table>

### **Ablation** Experimental

**Results of ablation experiments for hyperparameter *α* in boundary constraints:**

![image-20230512173406773](C:/Users/Administrator/AppData/Roaming/Typora/typora-user-images/image-20230512173406773.png)

**Ablation experimental results of each module of the proposed model."✓" means that the model adds this module, and "✘" means that the model removes this module:**

![image-20230512173420808](C:/Users/Administrator/AppData/Roaming/Typora/typora-user-images/image-20230512173420808.png)

**Comparison of model accuracy and convergence speed of mIoU training.Figure (a) with training accuracy on the left and Figure (b) with the convergence of sum and mIoU.**

![er_miou](../../../BMVC会议/images/er_miou.png)
