# LSTMSpeechTherapy
Trained an LSTM (keras, TensorFlow; python) on famous speeches delivered throughout history in order to generate conglomerated pseudo-speeches.

SpeechParsing.ipynb crawls and extracts top speeches from http://www.artofmanliness.com/2008/08/01/the-35-greatest-speeches-in-history/, and places the text content into appropriate files. 

SpeechLSTM.ipynb trains a multi-layer LSTM artificial NN on these speeches. The best fit weights are saved to an hdf5 file.
These weights can be loaded into the LSTM in order to output artificial speech text that incorporates patterns from world famous orators who lived in vastly different periods of time.
