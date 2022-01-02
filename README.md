# JudeasRX

![JudeasRx screenshot](images/JudeasRx-screenshot.jpg)

## Instructions
* Read the references given in the **Theory and Notation** section 
below
* Fire up the Jupyter Notebook judeas-rx.ipynb

The notebook draws the GUI (graphical user interface) shown in the above 
jpg. By playing with 
this GUI, you should be able to easily figure out how to use it. If something 
isn't clear,
read the FAQ file.

## Programming Aspects

The above jpg shows the entire GUI. There aren’t 
any other windows to the app. It’s a single window app. The app is written in Python and runs inside a Jupyter notebook. The controls that you see are coded using ipywidgets. The bar graph changes as the sliders move.


## UseCase 
The app considers the simple yet illustrative case of trying to prescribe a medicine differently for male versus female patients.  The Rx is based on probabilities called PNS, PN and PS that were invented by Pearl.

To do an Rx, one first allows the patients to take or not take the drug, according to their own discretion. Then one conducts a survey in which one collects info about who did or did not take the drug and whether they lived or died. This is called Observational Data (OD). Even though OD can be confounded, it serves to impose bounds on PNS, PN and PS. After the OD stage is completed, one can follow it up with a RCT (Randomized Control Experiment). In a RCT, patients must take or not take the drug as ordered by the doctor; no insubordination is tolerated. From the RCT, one collects Experimental Data (ED). The ED imposes tighter bounds on PNS, PN and PS than the bounds imposed by the OD alone.

## Mathematical Theory and Notation 
The mathematical theory and notation are described in gory  but 
entertaining detail in the chapter entitled  “Personalized Treatment 
Effects” of my book Bayesuvius. That chapter is totally based on a paper by 
Tian and Pearl. My only contribution(?) was to change the notation to a 
more “personalized” notation. [Bayesuvius](https://qbnets.wordpress.com/2020/11/30/my-free-book-bayesuvius-on-bayesian-networks/) is my free, 
open source book (about 560 pages so far) on Bayesian Networks and Causal 
Inference.

