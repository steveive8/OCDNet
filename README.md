# OCDNet

Created by Steve Ive

## OCDNet: Obsessive Convolutional Desne Net
### Neural Net consisting of parallel convolutional layers with Batch Expanding.

---
### Architecture

OCDNet uses parataxis CNN architecture, which the Convolution Layers are layed parallelly like as a node of Linear Layer. With the parallel architecture, batch is expanding as amount of ***Amount(prev layer's convolution nodes) X Amount(current layer's convolution nodes)***

To maximize the expanding batch's advantage and fix the batch size, OCDNet uses 3 ways.

- Add all the Batches and multiply with ***Expanding Gamma***
- ***Random Batching***
- Just add all the batches

---

### Arxiv

---

## How to use OCDNet

---

### Dataset: Stanford Car Dataset

https://www.kaggle.com/jutrera/stanford-car-dataset-images-in-224x224

---

#### How to set dataset directory

stanford-car/

ㄴ car_data/

ㄴ anno_test.csv

ㄴ names.csv

---

### Requirements

```pip install -r requirements.txt```
