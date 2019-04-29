# Neural Network Style Transfer

This is an implementation for [A Neural Algorithm of Artistic Style by Gatys, Ecker, and Bethge](https://arxiv.org/abs/1508.06576). It aims to take a base content image and base style image to make a third image which represents the content of the content image in the style of the style image.

For example, using these two images as content and style respectively:

![Content](https://github.com/jmontag43/neural_art_style/blob/master/images/waterways.jpg "Content Image")
![Style](https://github.com/jmontag43/neural_art_style/blob/master/images/starry_night.jpg "Style Image")

You can produce the following output image:

![Output](https://github.com/jmontag43/neural_art_style/blob/master/new_images/waterways_starry.jpg "Output Image")


## Requirements

- [Python 3.7](https://www.python.org/downloads/release/python-367/) (3.4, 3.5, and 3.6 should also work with TensorFlow 1.13 but are untested with this project)
- [TensorFlow 1.13.1](https://www.tensorflow.org/install) and [Pillow 6.0.0](https://pypi.org/project/Pillow/)
- (Optional for GPU support) An Nvidia GPU with [drivers >= 410.x](https://www.nvidia.com/Download/index.aspx?lang=en-us), [CUDA 10.0](https://developer.nvidia.com/cuda-zone), [CUPTI](https://docs.nvidia.com/cuda/cupti/) (ships with CUDA Tooklkit), [cuDNN >=  7.4.1](https://developer.nvidia.com/cudnn)

## Running the Project

Make sure you have the above requirements installed and pass `style_transfer.py` to your interpreter. An example of what this would look like on a generic Linux system is below.

It is recommended to use a virtual environment like [Virtualenv](https://virtualenv.pypa.io/en/stable/):
```sh
$ git clone https://github.com/jmontag43/neural_art_style.git
$ sudo pip install virtualenv
$ virtualenv myenv
$ source myenv/bin/activate
(myenv) $ cd neural_art_style
(myenv) $ pip install -r requirements.txt       # requirements-gpu.txt for gpu; requires above CUDA packages
(myenv) $ python style_transfer.py
```
**Note: Some shells (e.g. fish) require a different source command. In the fish example: `. myenv/bin/activate.fish`**

It is generally recommended to run this project with the GPU version; it takes me about 50 seconds to run 1000 iterations on my 2080ti and 205 seconds for **100** iterations on my CPU. However, the prebuilt TensorFlow GPU packages in PyPI often don't keep up to date with the CUDA packages (CUDA 10.1 came out 2 days after the prebuilt TensorFlow 1.13 used 10.0, making it almost instantly out of date). As a result, it's up to you if you'd like to run an outdated CUDA, a TensorFlow package managed by somebody who keeps up to date with your package manager, or build your own TensorFlow from source. The official TensorFlow GPU installation guide can be found [here](https://www.tensorflow.org/install/gpu).
