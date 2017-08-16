#cell 0
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils




#cell 1
# Read in speeches

file = open('filenames.txt')
fnames = []
for line in file:
    line = line.rsplit('\n')
    fnames.append(line[0])
file.close()

directory = 'speeches'
raw_txt = ''
for fname in fnames:
    fn = directory + '/' + fname + '.txt'
    f = open(fn,'r')
    txt = f.read()
    txt = txt.replace('Brett & Kate McKay','')
    txt = txt.replace('July 31, 2008','')
    raw_txt = raw_txt + txt
raw_txt = raw_txt.lower() # send to lower case
whitelist = set('abcdefghijklmnopqrstuvwxyz 1234567890.,;-\n')
raw_txt = ''.join(filter(whitelist.__contains__,raw_txt))


#cell 2
# Get 'vocabulary' that the machine has to learn. In addition, create a dictionary that maps chars to ints
chars = sorted(list(set(raw_txt)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
n_chars = len(raw_txt)
n_vocab = len(chars)
print("Total characteres: ", n_chars)
print("Total vocab: ", n_vocab)


#cell 3
# Extract sequence patterns. LSTM will be solving f([a1,a2,...,an]) = b1 
# , where each ai is a character and b1 is the output character
seq_length = 128
dataX = []
dataY = []
for i in range(0, n_chars-seq_length,1):
    seq_in = raw_txt[i:i+seq_length]
    seq_out = raw_txt[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total patterns:", n_patterns)

#cell 4
# Reshape patterns into numpy 3D matrix
X = numpy.reshape(dataX,(n_patterns,seq_length,1))
X = X / float(n_vocab)
# hot-encoding for output. In the algorithm, output will be determined based on max probability
y = np_utils.to_categorical(dataY)


#cell 5
# Create the LSTM model
model = Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')


#cell 6
filepath="weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks_list = [checkpoint]

#cell 7
model.fit(X,y,epochs=50,batch_size=64, callbacks=callbacks_list)

#cell 8
# Load in network with newly trained weights
filename = "weights-improvement-02-2.3400.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')
int_to_char = dict((i,c) for i,c in enumerate(chars))

#cell 9
import sys

# Lets generate some sentences 
start = numpy.random.randint(0,len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"",''.join([int_to_char[value] for value in pattern]),"\"")
for i in range(0,1000):
    x = numpy.reshape(pattern, (1, len(pattern),1))
    x = x / float(n_vocab)
    prediction = model.predict(x,verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone!")

#cell 10


