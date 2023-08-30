# Drug repositioning based on weighted local information augmented graph neural network
This is our tensorflow implementation of DRAGNN for drug repositioning associated with:

>**Drug repositioning based on weighted local information augmented graph neural network**  
>Yajie Meng, Yi Wang, Junlin Xu, Changcheng Lu, Xianfang Tang, Bengong Zhang, Geng Tian and Jialiang Yang 
 
## Environment Requirement
- tensorflow-gpu == 1.13.1
- keras == 2.2.4
- scikit-learn == 0.22.2
## Datasets
- Fdataset and Cdataset https://github.com/BioinformaticsCSU/BNNR
- LRSSL https://github.com/linwang1982/DRIMC
## Usage
- Please set the mode parameter to "cv," "case," and "analysis" in main.py to reproduce the 10-fold cross-validation results, case study results, and network analysis prediction results reported in our paper, respectively. When mode is specified as "case," please provide the desired "specific_name" to identify the specific case for the analysis.
- For "case" and "analysis", we strongly recommend running on Fdataset using CPU to ensure accurate reproducibility.
### **If you find our codes helpful, please kindly cite the following paper. Thanks!**
	@article{DRAGNN,
	  title={Drug repositioning based on weighted local information augmented graph neural network},
	  author={Yajie Meng, Yi Wang, Junlin Xu, Changcheng Lu, Xianfang Tang, Bengong Zhang, Geng Tian and Jialiang Yang},
	  year={2023},
	}
