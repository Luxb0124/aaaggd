## Official code implementation based on pytorch for paper.
### The framework of our proposed method.
![](./svg_imgs/01_framework.svg)
### Experiments
#### Visualization results comparison between state-of-the-art comparative methods and our AGDiffuser on MCGAN-Dataset.
![](./svg_imgs/03_exp_MCGAN.svg)
#### Comparison of visualization results between state-of-the-art methods and our AGDiffuser in terms of style learning and generative abilities on the Chinese100-Dataset
![](./svg_imgs/04_exp_c_style.svg)
#### Comparison of visualization results between state-of-the-art methods and our AGDiffuser in terms of shape learning and generative abilities on the Chinese100-Dataset
![](./svg_imgs/05_exp_c_shape.svg)
#### Visualization results of comprehensive shape and style tests in Chinese100-Data. The generated results of competitors often suffer from issues such as the presence of artifacts and a lack of refinement. In contrast, our method has high fidelity and is more in line with expectations.
![](./svg_imgs/06_exp_c_comp.svg)
### Run
#### MCGAN-Dataset
`python 20250302_CISGanDiff_Eng.py`
#### Chinese100-Dataset
`python 20250303_CISGanDiff_CHN.py`