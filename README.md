# tag-prediction
This is a repository for several tag prediction methods which can be compared against each other to find the best performing one for your individual purpose.
## Setup
* Install *daru*, *daru-io*, and *stopwords-filter*
* Create a *Daru::DataFrame* or import from json, csv, etc. Your data frame with training data must have a 'content' vector containing the text entries, and a 'tags' vector containing tags for training.
## Word Frequency Model
The word frequency model takes a data frame with (at least) content and training tags, and adds a 'predictions' vector to the data frame containing predicted tags for comparison to the original ones.
