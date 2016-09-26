import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from jinja2 import Template
from weasyprint import HTML

from data import Data, getFeaturesInfo


def createReport(features, filename):
  """
  """
  with open(templatepath) as f:
    template = Template(f.read())

  html = template.render(features=features)
  with open(filename + ".html", 'w') as f:
    f.write(html)

  HTML(string=html,
       base_url="").write_pdf(filename + ".pdf")

  

if __name__ == "__main__":
  basepath = "/home/alvaro/kaggle/house_prediction/"
  datapath = basepath + "data/"
  rawdatapath = datapath + "raw/"
  reportpath = basepath + "report/"
  imagespath = reportpath + "images/"
  templatepath = reportpath + "templates/report.html"
  outputpath = reportpath + "report"

  print "Loading data..."
  data = Data(rawdatapath)
  train = data.train
  print "Loading features info..."
  features = getFeaturesInfo(rawdatapath + "data_description.txt")
  features = {name: info for name, info in features.items() if name in train.columns}
  
  # Categorical features
  print "Reporting over categorical features"
  categoricalFeatures = [name for name, info in features.items()
                         if info["type"] == "categorical"]
  catTrain = train[categoricalFeatures]
  for name in catTrain:
    filename = imagespath + name + ".png"
    if os.path.exists(filename):
      features[name]["image"] = filename
      continue
    feature = catTrain[name]
    fig = plt.figure()
    feature.value_counts(filename).plot(title=name, kind='bar')
    fig.savefig(filename)
    plt.close(fig)
    features[name]["image"] = filename
    
  # Numerical features
  print "Reporting over numerical features"
  numericalFeatures = [name for name, info in features.items()
                       if info["type"] == "numerical"]
  numTrain = train[numericalFeatures]
  for name in numTrain:
    filename = imagespath + name + ".png"
    if os.path.exists(filename):
      features[name]["image"] = filename
      continue
    feature = numTrain[name]
    fig = plt.figure()
    hist = feature.hist(bins=50)
    plt.title(name)
    fig.savefig(filename)
    plt.close(fig)
    features[name]["image"] = filename

  createReport(features, outputpath)
