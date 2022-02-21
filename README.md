# Redol Classifier

A novel classification model trained to distinguish correctly from incorrectly labeled instances. The proposed method transforms any classification problem from a task of producing the correct label to a task of deciding if a given label is correct. To do so, a new dataset is created in which an attribute is added with a randomized label suggestion for each instance. The model then learns if the suggested labels are correct or not. The method builds an ensemble of decision trees trained on different randomized versions of this classification task.