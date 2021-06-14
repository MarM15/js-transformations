# Statically Detecting JavaScript Obfuscation and Minification Techniques in the Wild

This repository contains the code for the DSN'21 paper: "Statically Detecting JavaScript Obfuscation and Minification Techniques in the Wild".  
Please note that in its current state, the code is a Poc and not a fully-fledged production-ready API.


## Summary
Both malicious and benign JavaScript code can be *transformed*, i.e., obfuscated or minified, to, e.g., make reverse-engineering harder or improve website performance by reducing code size.  
To detect if a given JavaScript sample has been transformed, we propose a multi-task classifier, which leverages features extracted from the Abstract Syntax Tree (AST), which we enhance with control and data flow information. For scripts flagged as transformed, we subsequently propose a second layer of classification, which leverages traces left in the code syntax to recognize specific transformation techniques.


## Setup

```
install python3  # (tested with versions 3.6.7 and 3.7.4)
install python3-pip

install nodejs
install npm
cd src/pdg_generation/
npm install esprima  # (tested with version 4.0.1)
cd ../..

pip3 install -U scikit-learn # tested with version 0.23.2
pip3 install graphviz # tested with version 0.13.2

git clone https://github.com/Aurore54F/JStap
cd JStap/
pip3 install -r requirements.txt # (tested versions indicated in requirements.txt)
cd pdg_generation
npm install escodegen # (tested with 1.9.1)
cd ../classification
npm install esprima # (tested with 4.0.1)
cd ../..
```


## Usage
The first step is to train the models for level one and two. 
Level one creates the first vector space to analyze samples with the aim of distinguishing regular from transformed (minified or obfuscated) JavaScript code. Level two creates the second vector space for samples reported as transformed to detect specific transformation techniques.

**learner.py** automatically computes the features of given JavaScript files and subsequently trains the models.
Afterward, the models can be used to classify (unknown) Javascript files.
For this purpose, **classifier.py** can be used for two use cases, either to evaluate the accuracy of the models or to classify unlabeled samples.

By default, **learner.py** and **classifier.py** utilize one thread. If you wish to speed up the computation, you can specify the number of threads to be used with "-t".
Both can be executed with the following arguments:
```
required arguments:
  -i INPUT, --input INPUT
                        Points to a .txt with folders of JS-files (+ labels if not -p)

optional arguments:
  -h, --help            show this help message and exit
  -p, --predict         Stores the prediction for each JS-file specified with -i in results.txt
  -m MODEL, --model MODEL
                        The Path where the model is stored (default: ./model)
  -t THREADS, --threads THREADS
                        Number of threads (default: 1)
  -l1, --level1         Only evaluate the classifiers at level one
  --translate TRANSLATE
                        Points to a .txt to translate integer-labels to names, if -p is chosen
  --move MOVE           Moves successfully processed files to given folder
  --error ERROR         Creates an errors.txt in the specified path to log errors
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Sets the logging level
  --pdg_regenerate      Deletes already generated PDGs and generates them again. Use this if you changed your
                        trainingset
```

### learner.py
**learner.py** can be executed from the root directory of this project like below with the minimum of required arguments. 
It will read the files from the folders specified with "-i", assign the corresponding labels, calculate the features, and train the models.

#### Example:
```
python3 src/learner.py -i helper.txt
```
The text file specified with "-i" has to be in the format as below.
##### helper.txt:
```
# Level 1
/PATH_TO_TRAININGSET/normal;0
/PATH_TO_TRAININGSET/minified;1
/PATH_TO_TRAININGSET/obfuscated;2
# Level 2
/PATH_TO_TRAININGSET/transformationtechnique-1;21
/PATH_TO_TRAININGSET/transformationtechnique-2;22
/PATH_TO_TRAININGSET/transformationtechnique-3;23
...
```
For every line, the files in the specified directory will be assigned to the labels after the ";". It is also possible to assign multiple labels to one folder, separated by a ",". All labels for level 2 have to start with a "2".

### classifier.py
**classifier.py** can also be executed from the root directory of this project like below with the minimum of required arguments.
It has two use cases. By default, it will evaluate the accuracy on unknown labeled JavaScript files.
With the additional argument "-p", it will instead store the prediction of unknown unlabeled JavaScript files.

#### Example-evaluation:
```
python3 src/classifier.py -i helper.txt
```
##### helper.txt:
```
# Level 1
/PATH_TO_EVALUATIONSET/normal;0
/PATH_TO_EVALUATIONSET/minified;1
/PATH_TO_EVALUATIONSET/obfuscated;2
# Level 2
/PATH_TO_EVALUATIONSET/transformationtechnique-1;21
/PATH_TO_EVALUATIONSET/transformationtechnique-2;22
/PATH_TO_EVALUATIONSET/transformationtechnique-3;23
```

#### Example-prediction:
```
python3 ./src/classifier.py -i ./helper.txt -p --translate ./translate.txt
```

##### helper.txt
```
/PATH_TO_SAMPLESET/Files-to-be-classified
```
##### translate.txt
```
normal;0
minified;1
obfuscated;2
transformationtechnique-1;21
transformationtechnique-2;22
transformationtechnique-3;23
```


## Cite this work
If you use our tool for academic research, you are highly encouraged to cite our DSN'21 paper: "Statically Detecting JavaScript Obfuscation and Minification Techniques in the Wild":
```
@inproceedings{moog2021,
    author = {Marvin Moog and Markus Demmel and Michael Backes and Aurore Fass},
    title = "{Statically Detecting JavaScript Obfuscation and Minification Techniques in the Wild}",
    booktitle = {Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)},
    year = {2021}
}
```

### Abstract:

JavaScript is both a popular client-side programming language and an attack vector. While malware developers transform their JavaScript code to hide its malicious intent and impede detection, well-intentioned developers also transform their code to, e.g., optimize website performance.

In this paper, we conduct an in-depth study of code transformations in the wild. Specifically, we perform a static analysis of JavaScript files to build their Abstract Syntax Tree (AST), which we extend with control and data flows. Subsequently, we define two classifiers, benefitting from AST-based features, to detect transformed samples along with specific transformation techniques. Besides malicious samples, we find that transforming code is increasingly popular on Node.js libraries and client-side JavaScript, with, e.g., 90% of Alexa Top 10k websites containing a transformed script. This way, code transformations are no indicator of maliciousness. Finally, we showcase that benign code transformation techniques and their frequency both differ from the prevalent malicious ones.


## License

This project is licensed under the terms of the AGPL3 license, which you can find in ```LICENSE```.
