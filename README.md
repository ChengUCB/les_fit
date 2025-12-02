# les_fit

## Summary 
We present **LES (Latent Ewald Summation)** ([https://github.com/ChengUCB/les](https://github.com/ChengUCB/les)) as a plug-in library designed to add long-range interactions to short-range machine learning interatomic potentials (MLIPs). 

This repository contains the data and scripts used in the study [*A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials*](https://arxiv.org/abs/2507.14302).
Here we demonstrate its integration with MLIPs such as **MACE**, **NequIP**, **Allegro**, **CACE**, and **CHGNet**, and provide training scripts and trained models. 
In particular, we provide **MACELES-OFF** model trained on the SPICE dataset using the the [**MACE** package](https://github.com/ChengUCB/mace) developed by the Cheng group, while the [**converted MACELES-OFF**](https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/MACELES-OFF_small_converted.model) model is compatible with the current [**MACE** main branch](https://github.com/ACEsuit/mace).

Here you can find MLIP packages **with LES implementation**.
| Package | Link |
|---------|------|
| **CACE**   | [github.com/BingqingCheng/cace](https://github.com/BingqingCheng/cace) |
| **MACE**   | [github.com/ChengUCB/mace](https://github.com/ChengUCB/mace) |
| **MACE(updated)**   | [github.com/ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **NequIP** | [github.com/ChengUCB/NequIP-LES](https://github.com/ChengUCB/NequIP-LES) |
| **Allegro** | [github.com/ChengUCB/NequIP-LES](https://github.com/ChengUCB/NequIP-LES) |
| **MatGL**  | [github.com/ChengUCB/matgl](https://github.com/ChengUCB/matgl) |

## 📣 Update 
[2025-10] The **`MACELES`** model has been implemented in the main [**MACE** repository](https://github.com/ACEsuit/mace). Example training and evaluation scripts are available in `./MLIPs/MACE-LES-new`.

[2025-12] For the [**MACE** main branch](https://github.com/ACEsuit/mace), one needs to use the [**converted MACELES-OFF model**](https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/MACELES-OFF_small_converted.model) (MD5: cc10e937b55e09f05b16dba756e2311b).

## Usage 
Please refer to the specific folder for related scripts and trained MLIPs.


## License
This project is licensed under the CC BY-NC 4.0 License.

## Citation


```text
@article{Kim2025universal,
  title = {A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials},
  author = {Kim, Dongjin and Wang, Xiaoyu and Zhong, Peichen and King, Daniel S and Inizan, Theo Jaffrelot and Cheng, Bingqing},
  journal={arXiv preprint arXiv:2507.14302},
  year={2025}
}

@article{cheng2025latent,
  title={Latent Ewald summation for machine learning of long-range interactions},
  author={Cheng, Bingqing},
  journal={npj Computational Materials},
  volume={11},
  number={1},
  pages={80},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

@article{King2025Machine,
  title = {Machine Learning of Charges and Long-Range Interactions from Energies and Forces},
  author = {King, Daniel S. and Kim, Dongjin and Zhong, Peichen and Cheng, Bingqing},
  journal = {Nature Communications},
  volume = {16},
  number = {1},
  pages = {8763},
  year = {2025},
  publisher = {Nature Publishing Group},
  doi = {10.1038/s41467-025-63852-x},
}


@article{zhong2025machine,
  title={Machine learning interatomic potential can infer electrical response},
  author={Zhong, Peichen and Kim, Dongjin and King, Daniel S and Cheng, Bingqing},
  journal={arXiv preprint arXiv:2504.05169},
  year={2025}
}


```

## Contact

For any queries regarding LES, please contact Bingqing Cheng at tonicbq@gmail.com.
