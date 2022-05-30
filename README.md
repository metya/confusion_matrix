# Confusion Matrix for Object Detection and Instance Segmentation

That's my implementation of class aware confusion matrix for object detection and instance segmentations. Particulary it uses COCO format of datasets for targets and predictions. But it easily can be rewrited to another format type. Also it uses pytorch for typings, but again easily can be replaces with tensorflow for example.

### Uses

```python
>>> from confusion_matrix import ConfusionMatrix

>>> confusion_matrix = ConfusionMatrix(class_names={0: 'class1', 1: 'class2'},
...                                            thrs_config={0: 0.5, 1: 0.5})
>>> for images, targets in test_dataloader:
>>>     outputs = model(images)
>>>     confusion_matrix.update(targets, outputs)

>>> confusion_matrix.plot(show=True)

or 

>>> confusion_matrix.pretty_plot()
```

### Instalation

```bash
pip install git+https://github.com/metya/confusion_matrix
```