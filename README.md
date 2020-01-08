
# :notebook_with_decorative_cover:  Usage
computer vision final project

## Download dataset
In this demo, we will use CASIA-WebFace for training, then we use LFW and YTF for testing.
- Download [CASIA-WebFace](https://drive.google.com/file/d/1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz/view)
- Download [LFW](http://vis-www.cs.umass.edu/lfw/)
- Download [YTF](https://www.cs.tau.ac.il/~wolf/ytfaces/)


## Preprocessing 
1. Download the original images of CASIA-WebFace dataset and align the images with the following command:
- Please unzip `data/ldmark_casia_mtcnncaffe.rar` before run the below command.
```
python align/align_dataset.py data/ldmark_casia_mtcnncaffe.txt \
data/casia_mtcnncaffe_aligned \
--prefix /path/to/CASIA-Webface/images \
--transpose_input --image_size 96 112
```
- To get the image path list for training, enter the` /data` directory, and run the notebook `write_path.ipynb`. You will get `list_casia.txt` under `/data` directory.

2. Download the original images of LFW dataset and align the images with the following command:
```
python align/align_dataset.py data/ldmark_lfw_mtcnncaffe.txt \
data/lfw_mtcnncaffe_aligned \
--prefix /path/to/LFW/images \
--transpose_input --image_size 96 112
```
3. Download the aligned image of YTF from above url.
( aligned_images_DB.tar.gz )

## Training 
1. The base embedding network(Spherenet) can be download from [here](https://drive.google.com/file/d/1MiC_qCj5GFidWLtON9ekClOCJu6dPHT4/view). Please put the file `sphere64_caisa_am` under the directory `pretrained/`.
2. The configuration file `sphere64_casia.py` for training is under `config/`folder, where you can define the training data, pre-trained model, and other hyper-parameters.
3. The uncertainty module that we are going to train is in `models/uncertainty_module.py`.
4. Use the following command to train the model:
```
python train.py config/sphere64_casia.py
```
- The command will create an folder under`log/improve_PFE/`, which saves all the checkpoints and summaries. Then you can evaluate the method's performance by restoring the training model.
## Testing
- We already put the pretrained model in [google dirve](https://drive.google.com/drive/folders/1S2hLByvH51W301nZXnHzeevrWslDAoew?usp=sharing). You can skip the train steps, using exactly the test command to get the result.


### YTF dataset
- Ours method
**Please download the [logs](https://drive.google.com/file/d/1IC32GHreD6nVcRSQ2B3Kbg0PzM-5SIEw/view?usp=sharing) and put it under  `eva_log/`directory.**
```
python eval_ytf.py --model_dir eva_log/our_method \
--dataset_path /path/to/your/aligned_data/directory \ 
--protocol_path  proto/ytf_pairs.txt
```

- PFE method
**Pleas download the [pretrained model](https://drive.google.com/file/d/1BeTUYnc__u1_cYEKoXqfGDQjdk2TChoD/view) and put `PFE_sphere64_casia_am` file under `pretrained/`.**

```
python eval_ytf.py --model_dir pretrained/PFE_sphere64_casia_am \
--dataset_path /path/to/your/aligned_data/directory \ 
--protocol_path  proto/ytf_pairs.txt
```

### LFW dataset
- Ours method
**Please download the [logs](https://drive.google.com/file/d/16kkJr00nA6X5ALOm_MBGyAwJyLhQgi_Z/view?usp=sharing) and put it under  `eva_log/`directory.**
```
python eval_lfw.py --model_dir eva_log/our_methods \
--dataset_path data/lfw_mtcnncaffe_aligned
```

- PFE method
 
 ```
python eval_lfw.py --model_dir pretrained/PFE_sphere64_casia_am \ 
--dataset_path data/lfw_mtcnncaffe_aligned
```

## :bar_chart: Test result 
- The test results are obtained using exactly this demo code.
- The baseline result is get from PFE paper github which is a base embedding network method.

| Method   |   LFW    |    YTF   |
| -------- | -------- | -------- |
| Ours     | 87.02    |  99.483  |
| PFE      | 86.52    | 99.467   |
| Baseline | None     |  99.2    |

## Reference
```
@article{shi2019PFE,
  title = {Probabilistic Face Embeddings},
  author = {Shi, Yichun and Jain, Anil K.},
  booktitle = {arXiv:1904.09658},
  year = {2019}
}
```
