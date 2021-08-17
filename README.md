# OCDNet

Created by Steve Ive

## OCDNet: Obsessive Convolutional Desne Net
### Neural Net consisting of parallel convolutional layers with Batch Expanding.

Model for large scale image classification. Uses parallel architecture that expanding Batches by each Convolutional Desne Layer.

---
### Architecture

OCDNet uses parataxis CNN architecture, which the Convolution Layers are layed parallelly like as a node of Linear Layer. With the parallel architecture, batch is expanding as amount of **Amount(prev layer's convolution node) X Amount(current layer's convolution node)**

To maximize the expanding batch's advantage and fix the batch size, OCDNet uses 3 ways.

- Add all the Batches and multiply with ***Expanding Gamma***
- ***Random Batching***
- Just add all the batches

---

### Requirements

```pip install -r requirements.txt```
