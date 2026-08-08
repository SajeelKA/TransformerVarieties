import numpy as np
from datetime import datetime
import requests

wordToTensor = lambda sInput, tokenMapping: [tokenMapping[letter] for letter in sInput] 
tensorToWord = lambda sIndexes, tokenMapping: [list(tokenMapping.keys())[i.item()] for i in sIndexes]


    
def tinyDataLoader(dataSet, batchSize, blockSize):
  batch = []
  toPredict = []
  i = 0
  block = 0
  toShuffle = np.arange(0,batchSize)
  np.random.shuffle(toShuffle)

  for b in range(0, len(dataSet)//blockSize):
    batch.append(dataSet[block : (block + blockSize)])
    toPredict.append(dataSet[block + 1 : (block + 1 + blockSize)]) #adding next letter for the one to predict
    block += blockSize
    if (b+1) % batchSize == 0:
      # yield batch, toPredict
      yield np.array(batch)[toShuffle], np.array(toPredict)[toShuffle]
      batch = []
      toPredict = []

def getRawData():
  data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
  data = requests.get(data_url).text

  # get all the unique characters that occur in this text
  possibleTokens = sorted(list(set(data)))

  return data, possibleTokens, len(possibleTokens)
