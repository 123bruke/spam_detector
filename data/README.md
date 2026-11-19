# Dataset files

`demo_sms.csv` contains 100 **synthetic** messages (50 ham-like and 50 spam-like) generated only to smoke-test the training pipeline without an internet connection. It is deliberately simple and balanced, so its evaluation scores are not representative of real-world spam detection.

For a meaningful experiment, run `python train.py` while connected to the internet. The loader will download the UCI SMS Spam Collection into this folder as `SMSSpamCollection` and use that dataset for training.

UCI dataset: https://archive.ics.uci.edu/dataset/228/sms
Citation: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
License: CC BY 4.0.
