# MDAT
## Boosting SAR Aircraft Detection Performance with Multi-Stage Domain Adaption Training

This is the reposity of the research paper "Boosting SAR Aircraft Detection Performance with Multi-Stage Domain Adaption Training", which is developed based on the [MMDetecion](https://github.com/open-mmlab/mmdetection) framework. 

## Abstract
Deep learning has achieved significant success in various synthetic aperture radar (SAR) imagery interpretation tasks. However, automatic aircraft detection is still challenging due to the high la-beling cost and limited data quantity. To address this issue, we propose a multi-stage domain adaption training framework to efficiently transfer the knowledge from optical imagery and boost SAR aircraft detection performance. For overcomingTo overcome the significant domain discrep-ancy between optical and SAR images, the training process can be divided into three stages: image translation, domain adaptive pretraining, and domain adaptive finetuning. First, CycleGAN is used to translate optical images into SAR-style images and reduce global-level image divergence. Next, we propose multilayer feature alignment to further reduce the local-level feature distribution distance. By domain adversarial learning in both pretrain and finetune stages, the detector can learn to extract domain-invariant features that are beneficial to the learning of generic aircraft character-istics. To evaluate the proposed method, extensive experiments are conducted on a self-built SAR aircraft detection dataset. The results indicate that by using the proposed training framework, the average precision of Faster RCNN gains an increase of 2.4 and that of YOLOv3 is improved by 2.6, which outperforms other domain adaption methods. By reducing the domain discrepancy between optical and SAR in three progressive stages, the proposed method can effectively mitigate the domain shift, thereby enhancing the efficiency of knowledge transfer. It greatly improves the de-tection performance of aircraft and offers an effective approach to address the limited training data problem of SAR aircraft detection.

Codes will be released soon.
