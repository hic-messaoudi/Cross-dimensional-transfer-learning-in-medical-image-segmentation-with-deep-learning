# Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning

The repository contains the official implementation of the architectural models described in [Cross-dimensional transfer learning in medical image segmentation with deep learning](https://doi.org/10.1016/j.media.2023.102868).

![graph](https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/assets/83643719/a1c0fa68-57e7-4f9d-8a7e-ad20483c4f9a)


:star: **Omnia-Net**, one of the architectures, which can utilize any pre-trained 2D classification network as an encoder within a U-Net-like structure. This approach proves effective in both 2D and 3D settings, exhibiting consistent performance across different medical imaging modalities.

📦 **Omnia-Net Model Availability**

The pytorch implementation of the network is available [HERE](https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/tree/main/Models/CAMUS) for the version b0 and [HERE](https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/tree/main/Models/CHAOS) for the version b4.

![2d_chaos](https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/assets/83643719/d999798e-d40a-4968-9095-cae75b000d66)


:star: **DS-Net** (Dimensionally-Stacked Network), is another architecture presented in the article, which comprises a randomly initialized 3D encoder, Omnia-Net (2D), and a 3D decoder. DS-Net leverages the strengths of both 2D and 3D representations to enhance the segmentation process.

📦 **DS-Net Model Availability**

The pytorch implementation of the network is available [HERE](https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/tree/main/Models/BraTS/DS-Net)

![brats](https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/assets/83643719/55a7069c-0c51-4d75-9c2d-8f29dd93c6af)


:star: The final architecture is **DX-Net** (Dimensionally-eXpanded Network), which employs a sequence of 3D weights extrapolated from the weights of a pre-trained 2D encoder. These 3D weights initialize the encoder component of a 3D U-Net-like architecture. By leveraging the pre-trained 2D weights, DX-Net aims to enhance the performance of the 3D encoder for faster and more accurate 3D medical image segmentation.

📦 **DX-Net Model Availability**

The pytorch implementation of the network is available [HERE](https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/tree/main/Models/BraTS/DX-Net)

![3d_scores](https://github.com/hic-messaoudi/Cross-dimensional-transfer-learning-in-medical-image-segmentation-with-deep-learning/assets/83643719/62b1048f-2888-4e96-b8a9-f00438fa50e3)


Overall, this repository provides the necessary codebase for replicating the experiments and results presented in the research article, enabling further exploration and development in the field of cross-dimensional transfer learning for medical image segmentation using deep learning techniques.

:books: Citing
```
@article{MESSAOUDI2023102868,
title = {Cross-dimensional transfer learning in medical image segmentation with deep learning},
journal = {Medical Image Analysis},
volume = {88},
pages = {102868},
year = {2023},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2023.102868},
url = {https://www.sciencedirect.com/science/article/pii/S1361841523001287},
author = {Hicham Messaoudi and Ahror Belaid and Douraied {Ben Salem} and Pierre-Henri Conze},
keywords = {Medical image segmentation, Transfer learning, Convolutional neural networks, Cross-dimensional transfer},
}
```

📚 References

1. Kavur, A. E., Gezer, N. S., Barış, M., Aslan, S., Conze, P.-H., Groza, V., Pham, D. D., Chatterjee, S., Ernst, P., Özkan, S., Baydar, B., Lachinov, D., Han, S., Pauli, J., Isensee, F., Perkonigg, M., Sathish, R., Rajan, R., Sheet, D., … Selver, M. A. (2021). CHAOS Challenge - combined (CT-MR) healthy abdominal organ segmentation. In Medical Image Analysis (Vol. 69, p. 101950). Elsevier BV. https://doi.org/10.1016/j.media.2020.101950

2. Messaoudi, H., Belaid, A., Allaoui, M. L., Zetout, A., Allili, M. S., Tliba, S., Ben Salem, D., & Conze, P.-H. (2021). Efficient Embedding Network for 3D Brain Tumor Segmentation. In Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (pp. 252–262). Springer International Publishing. https://doi.org/10.1007/978-3-030-72084-1_23
   
3. Pierre-Henri Conze, Sylvain Brochard, Val´erie Burdin, Frances T.Sheehan,and Christelle Pons. Healthy versus pathological learning transferability
in shoulder muscle MRI segmentation using deep convolutional encoderdecoders. Comput. Med. Imaging Graph., 83:101733, 2020. DOI: [10.1016/j.compmedimag.2020.101733.](https://doi.org/10.1016/j.compmedimag.2020.101733)
   
4. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, and J. Kirbyet al. Segmentation labels and radiomic features for the pre-operative scans
of the tcga-gbm collection. The Cancer Imaging Archive, 2017. DOI:10.7937/K9/TCIA.2017.KLXWJJ1Q.

5. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, and J.S. Kirby et al. Advancing the cancer genome atlas glioma mri collections with expert segmentation labels and radiomic features. Nature Scientific Data, page 4:170117, 2017. DOI:[10.1038/sdata.2017.117](https://doi.org/10.1038/sdata.2017.117).

6. S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, and A. Crimi et al. Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the brats challenge. arXiv preprintarXiv:1811.02629, 2018.
   
7. Ujjwal Baid, Satyam Ghodasara, Michel Bilello, et al (2021). The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification. CoRR, abs/2107.02314.
   
8. Wightman, R., Raw, N., Soare, A., Arora, A., Ha, C., Reich, C., Guan, F., Kaczmarzyk, J., MrT23, , Mike, SeeFun, Contrastive, Rizin, M., Hyeongchan Kim, Kertész, C., Dushyant Mehta, Cucurull, G., Kushajveer Singh, Hankyul, … Uchida, Y. (2023). rwightman/pytorch-image-models: v0.8.10dev0 Release (v0.8.10dev0) [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.4414861
  
