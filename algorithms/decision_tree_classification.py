class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        self.value = value              # for leaf node

class DecisionTreeClassifier():         # we use recursion to form the tree
    def __init__(self, min_sample_split=2, max_depth=2):
        self.root = None
        
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def buildTree(self, dataset, curr_depth=0):
        
        pass

    
    def fit(dataset):
        pass

    def predict(dataset):
        pass

    def __getEntropy(self):
        # Entropy formula: Entropy = -Σ (p_i * log2(p_i))
        pass

    def __getInformationGain(self):
        # Information Gain formula: IG = Entropy(Parent) - Weighted_Entropy(Children)
        pass

    def __getGiniIndex(self):
        # Gini Index formula: Gini = 1 - Σ (p_i^2)
        pass

if __name__ == "__main__":
    dt = DecisionTreeClassifier()
    
    pass
