from predict.featureextractor import FeatureExtractor

"""

    This class contains the benchmark predictor. For the training data, calculate the occurrence of each traffic
    sign and assign these occurrence probabilities to the new test data. Conclusion: it's really really bad,
    don't do worse!


"""

class BenchmarkPredictor(FeatureExtractor):

    def __init__(self):
        self.occurrenceProbabilities = {}

    def train(self, trainingData, results):
        counter = 0
        for image in trainingData:
            if results[counter] in self.occurrenceProbabilities:
                self.occurrenceProbabilities[results[counter]] += 1
            else:
                self.occurrenceProbabilities[results[counter]] = 0
            counter += 1

        for occurrenceProb in self.occurrenceProbabilities:
            self.occurrenceProbabilities[occurrenceProb] = self.occurrenceProbabilities[occurrenceProb]/counter

    def predict(self, image):
        return self.occurrenceProbabilities
