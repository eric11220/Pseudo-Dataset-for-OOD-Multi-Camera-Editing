## *Pseudo Dataset Generation for Out-of-Domain Multi-Camera View Recommendation*
----

This repository is an official implementation of *Pseudo Dataset Generation for Out-of-Domain Multi-Camera View Recommendation* in Pytorch (VCIP 2024)


## Abstract
Multi-camera systems are indispensable in movies, TV shows, and other media. Selecting the appropriate camera at every timestamp has a decisive impact on production quality and audience preferences. Learning-based view recommendation frameworks can assist professionals in decision-making. However, they often struggle outside of their training domains. The scarcity of labeled multi-camera view recommendation datasets exacerbates the issue. Based on the insight that many videos are edited from the original multi-camera videos, we propose transforming regular videos into pseudo-labeled multi-camera view recommendation datasets. Promisingly, by training the model on pseudo-labeled datasets stemming from videos in the target domain, we achieve a 68% relative improvement in the model's accuracy in the target domain and bridge the accuracy gap between in-domain and never-before-seen domains.

### Contribution
- We identify the poor domain generalizability of multicamera view recommendation models.
- We propose generating pseudo-labeled multi-camera editing datasets with regular videos to mitigate the lack of labeled data on an arbitrary domain.
- With the proposed pseudo-labeled multi-camera editing datasets, we achieve a 68% relative improvement in the model’s classification accuracy in the target domain.

## Requirement
Installing packages
```sh
pip install -r requirements.txt
```

## Pseudo Dataset Generation Given a Video or Directory of Videos
```
bash generate.sh video_path
```
This will
- Run shot boundary detection on all videos and write results for individual videos to `output/shot_bouundaries`.
- Create a JSON file `output/shots.json` that contains the start and end frames for every video shot.
- Generate the pseudo dataset in a JSON file `output/pseudo_data.json` and individual frames from each video in `output/frames`.


The format of `pseuudo_data.json` is as follows, adapted from [TVMCE](https://github.com/VirtualFilmStudio/TVMCE).
```
pseudo_data.json
{
  "data": [
    {                                                                                             # one single pseudo data instance
      "video_id": "gs-o7elkwe8.mp4",                                                              # video name
      "sampleInterval": 5,                                                                        # interval between historic frames
      "startFrame": 504,                                                                          # start frame
      "outputList": [504, 509, 514, 519, 524, 529, 534, 539, 544, 549, 554, 559, 564, 569, 574],  # historic frames
      "outputCam": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],                                 # pseudo camera of each historic frame
      "candidates": [4454, 0, 3897, 1301, 580, 4220],                                             # pseudo candidates and ground truth
      "selectCAM": 5,                                                                             # groundtruth
      "CAMList": [1, 2, 3, 4, 5, 6]                                                               # pseudo camera of each candidate and ground truth
    },
    ...
  ]
  "meta": [
    "gs-o7elkwe8.mp4": {     # meta infomation for each video
      "frame2cam": {         # which pseudo camera each frame is assigned to
        "0": 2,
        "5": 2,

        ...

        "4454": 1
      },
      "segments": [          # start and end frame for each detected shot
        [0, 579],
        [580, 1084],

        ...

        [4454, 4457]
      ]
    },
    ...
  ]
}
```

## Citation
If you find this paper/code helpful for your research, please consider citing us:

**Pseudo Dataset Generation for Out-of-Domain Multi-Camera View Recommendation**

```
@inproceedings{lee2024_multicam_recom,
  author={Lee, Kuan-Ying and Zhou, Qian and Nahrstedt, Klara},
  title={Pseudo Dataset Generation for Out-of-domain Multi-Camera View Recommendation},
  booktitle={IEEE Visual Communications and Image Processing (VCIP)}
  year={2024},
}
```

## Acknowledgement
This work is developed based on the multi-camera editing dataset collected by [TVMCE](https://github.com/VirtualFilmStudio/TVMCE).
