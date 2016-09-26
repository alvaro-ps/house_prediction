"""
Module to work with data
"""
import os
import logging

import pandas as pd

def getFeaturesInfo(data_description_file):
  """
  Returns a list of dictionaries each of them
  describing a feature.

  {
    "name": 
    "description":
    "type":
    [if type=categorical] "values": [
                                      {
                                        "code":
                                        "name":
                                      }
                                    ]

  }
  """
  featuresInfo = dict()
  with open(data_description_file) as f:
    line = f.next().strip()
    featureName, featureDescription = line.split(": ")
    featureValues = []
    for line in f:
      line = line.strip()
      if not line:
        continue
      if "\t" not in line:
        # store previous feature
        featureInfo = {
                        "description": featureDescription,
                        "name": featureName
                      }
        if featureValues:
          featureInfo["type"] = "categorical"
          featureInfo["values"] = featureValues
        else:
          featureInfo["type"] = "numerical"
        featuresInfo[featureName] = featureInfo

        # get new feature info
        featureName, featureDescription = line.split(": ")
        featureValues = []
      else:
        categoryCode, categoryName = line.split("\t")
        featureValues.append({
                               "code": categoryCode,
                               "name": categoryName
                             })
      
  return featuresInfo 

def loadDataset(path):
  """
  """
  data = pd.read_csv(path, index_col=0, na_values={'MasVnrArea': ["NA"]}, keep_default_na=False)
  return data

class Data():
  """
  Class to deal with the data set
  """
  def __init__(self, path):
    """
    path: folder where the raw data is stored
    """
    self.trainPath = os.path.join(path, "train.csv")
    self.testPath = os.path.join(path, "test.csv")

  def __getDataset(self, type, path):
    """
    """
    if not hasattr(self, type):
      logging.info("Loading data set...")
      setattr(self, type, loadDataset(path))
      
    return getattr(self, type)

  @property
  def train(self):
    """
    Return a data frame with the train set
    """
    return self.__getDataset('trainset', self.trainPath)

  @property
  def test(self):
    """
    Return a data frame with the test set
    """
    return self.__getDataset('trainset', self.testPath)

  def xytrain(self, yname='SalePrice'):
    """
    Returns a tuple (Xtrain, ytrain)
    """
    train = self.train
    y = train[yname]
    X = train.drop(yname, axis="columns")
    return X, y
