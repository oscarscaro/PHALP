# Tracking People by Predicting 3D Appearance, Location & Pose
Code repository for the paper "Tracking People by Predicting 3D Appearance, Location & Pose". [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zeHvAcvflsty2p9Hgr49-AhQjkSWCXjr?usp=sharing) \
[Jathushan Rajasegaran](http://people.eecs.berkeley.edu/~jathushan/), [Georgios Pavlakos](https://geopavlakos.github.io/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/). \
[(paper)](https://arxiv.org/abs/2112.04477)       [(project page)](https://people.eecs.berkeley.edu/~jathushan/PHALP/). 
 
This code repository provides a code implementation for our paper PHALP, with installation, preparing datasets, and evaluating on datasets, and a demo code to run on any youtube videos. 

<p align="center"><img src="./utils/imgs/teaser.png" width="800"></p>

**Abstract** : <em>In this paper, we present an approach for tracking people in monocular videos, by predicting their future 3D representations. To achieve this, we first lift people to 3D from a single frame in a robust way. This lifting includes information about the 3D pose of the person, his or her location in the 3D space, and the 3D appearance. As we track a person, we collect 3D observations over time in a tracklet representation. Given the 3D nature of our observations, we build temporal models for each one of the previous attributes. We use these models to predict the future state of the tracklet, including 3D location, 3D appearance, and 3D pose. For a future frame, we compute the similarity between the predicted state of a tracklet and the single frame observations in a probabilistic manner. Association is solved with simple Hungarian matching, and the matches are used to update the respective tracklets. We evaluate our approach on various benchmarks and report state-of-the-art results. </em> 

## Installation

We recommend creating a clean [conda](https://docs.conda.io/) environment and install all dependencies.
You can do this as follows:
```
conda env create -f _environment.yml
```

After the installation is complete you can activate the conda environment by running:
```
conda activate PHALP
```

Install PyOpenGL from this repository:
```
mkdir external
pip uninstall pyopengl
git clone https://github.com/mmatl/pyopengl.git external/pyopengl
pip install ./external/pyopengl

git clone https://github.com/JonathonLuiten/TrackEval external/TrackEval
pip install -r external/TrackEval/requirements.txt

git clone https://github.com/brjathu/pytube external/pytube
cd external/pytube/; python setup.py install; cd ../..

cd external/neural_renderer/; python setup.py install; cd ../..
```

Additionally, install [Detectron2](https://github.com/facebookresearch/detectron2) from the official repository, if you need to run demo code on a local machine. We provide detections inside the _DATA folder, so for running the tracker on posetrack or mupots, you do not need to install Detectron2.

## Download Weights and Data

Please download this folder and extract inside the main repository.

- [_DATA/](https://drive.google.com/uc?id=13XUwfHFsuyr14wCfjn_BPkBwx-uQ8woX)

or you can run the following command.
`curl -L -o '_DATA.zip' 'https://drive.google.com/uc?id=13XUwfHFsuyr14wCfjn_BPkBwx-uQ8woX&confirm=t'; unzip _DATA.zip`

Besides these files, you also need to download the [neutral *SMPL* model](http://smplify.is.tue.mpg.de). Please go to the website for the corresponding project and register to get access to the downloads section. Create a folder `_DATA/models/smpl/` and place the model there. Otherwise, you can also run:

`python3 utils/convert_smpl.py`
    
## Testing

Once the posetrack dataset is downloaded at "_DATA/posetrack/posetrack_data/", run the following command to run our tracker on all validation videos. 

`python demo.py --track_dataset posetrack`

## Evaluation

To evaluate the tracking performance on ID switches, MOTA, and IDF1 and HOTA metrics, please run the following command.

`python3 evaluate_PHALP.py out/Videos_results/results/ PHALP posetrack`

## Demo

Please run the following command to run our method on a youtube video. This will download the youtube video from a given ID, and extract frames, run Detectron2, run HMAR and finally run our tracker and renders the video.

`python3 demo.py --track_dataset demo`


## Results ([Project site](http://people.eecs.berkeley.edu/~jathushan/PHALP/))

We evaluated our method on PoseTrack, MuPoTs and AVA datasets. Our results show significant improvements over the state-of-the-art methods on person tracking. For more results please visit our [website](http://people.eecs.berkeley.edu/~jathushan/PHALP/).


<p align="center"><img src="./utils/imgs/PHALP_1.gif" width="800"></p>
<p align="center"><img src="./utils/imgs/PHALP_2.gif" width="800"></p>
<p align="center"><img src="./utils/imgs/PHALP_3.gif" width="800"></p>
<p align="center"><img src="./utils/imgs/PHALP_4.gif" width="800"></p>
<p align="center"><img src="./utils/imgs/PHALP_5.gif" width="800"></p>
<p align="center"><img src="./utils/imgs/PHALP_6.gif" width="800"></p>
<p align="center"><img src="./utils/imgs/PHALP_7.gif" width="800"></p>

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [deep sort](https://github.com/nwojke/deep_sort)
- [SMPL-X](https://github.com/vchoutas/smplx)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [SPIN](https://github.com/nkolot/SPIN)
- [VIBE](https://github.com/mkocabas/VIBE)
- [SMALST](https://github.com/silviazuffi/smalst)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)

## Contact
Jathushan Rajasegaran - jathushan@berkeley.edu or brjathu@gmail.com
<br/>
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/brjathu/PHALP/issues).
<br/>
Discussions, suggestions and questions are welcome!


## Citation
If you find this code useful for your research or the use data generated by our method, please consider citing the following paper:
```
@article{rajasegaran2021tracking,
  title={Tracking People by Predicting 3D Appearance, Location \& Pose},
  author={Rajasegaran, Jathushan and Pavlakos, Georgios and Kanazawa, Angjoo and Malik, Jitendra},
  journal={arXiv preprint arXiv:2112.04477},
  year={2021}
}

```
