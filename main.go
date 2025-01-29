package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	pkgKNN "github.com/infiniteCrank/gobotai/knn"
	pkgNN "github.com/infiniteCrank/gobotai/nn"
	pkgTFIDF "github.com/infiniteCrank/gobotai/tfidf"
)

func main() {
	// init the corpus and supporting data
	tfidf := initializeTFIDF()
	var inputQuery []string
	inputQuery = append(inputQuery, "How do I define a function in go?")
	queryVec := pkgTFIDF.NewTFIDF(inputQuery)
	queryVec.CalculateScores()

	// fmt.Printf("corpus scores: %+v \n query scores: %+v \n", tfidf.Scores, queryVec.Scores)

	// 1. format the data for training
	knnData := pkgKNN.CreateDataSet(tfidf.Corpus)

	// Prepare inputs & targets for NN training based on TF-IDF scores
	targets := pkgNN.PrepareTargetData(tfidf.Scores) // You need to implement this function
	if len(targets) == 0 {
		log.Fatal("Target data is empty.")
	}

	// Create the neural network
	layerSizes := []int{len(tfidf.Scores), 10, 1} // Example structure
	activations := []int{pkgNN.SigmoidActivation, pkgNN.SigmoidActivation}
	nn := pkgNN.NewNeuralNetwork(layerSizes, activations, 0.01, 0.01)

	// Train the network
	// Get the feature order based on the scores
	featureOrder := pkgNN.GetFeatureOrder(tfidf.Scores)

	// Prepare input for training
	trainingData := pkgNN.ConvertKNNDataToTrainingFormat(knnData.Dataset, featureOrder) // Convert KNN dataset
	if len(trainingData) == 0 {
		log.Fatal("Training data is empty.")
	}

	// Ensure trainingData and targets have the same number of samples
	if len(trainingData) != len(targets) {
		log.Fatalf("Number of training inputs does not match number of targets: %d vs %d", len(trainingData), len(targets))
	}

	// Train the neural network
	nn.Train(trainingData, targets, 1000, 0.95, 100)

	// My query TF-IDF scores
	queryVecScores := queryVec.Scores

	queryVector := pkgNN.ConvertScoresToVector(queryVecScores, featureOrder)

	// Now call Predict with the converted vector
	results := nn.Predict(queryVector)

	// Print the result
	fmt.Printf("Neural Network Relevance Scores for Query: %+v\n", results)

	// Retrieve the most relevant answer using KNN
	k := 21 // Number of neighbors to consider
	answers := pkgKNN.KNN(queryVecScores, knnData.Dataset, k, 3)

	// Print the result
	fmt.Printf("Most relevant content: %+v \n", answers)

	corpusKeywords := tfidf.ExtractKeywords(20)
	var relatedKeywords []string
	for term := range corpusKeywords {
		if strings.Contains(strings.ToLower(inputQuery[0]), term) {
			relatedKeywords = append(relatedKeywords, term)
		}
	}
	fmt.Printf("Related Keywords: %+v \n", relatedKeywords)

	var newQuery string
	for _, keyword := range relatedKeywords {
		newQuery += " " + keyword
	}
	var stringSlice []string
	stringSlice = append(stringSlice, newQuery)
	newQueryVec := pkgTFIDF.NewTFIDF(stringSlice)
	newQueryVec.CalculateScores()
	newQueryVecScores := newQueryVec.Scores
	newAnswers := pkgKNN.KNN(newQueryVecScores, knnData.Dataset, k, 3)
	fmt.Printf("Best content: %+v \n", newAnswers)
}

func initializeTFIDF() *pkgTFIDF.TFIDF {
	// Load the existing corpus of training phrases as text
	corpus, err := LoadCorpus("go_corpus.md")
	if err != nil {
		log.Fatal("Error loading corpus:", err)
	}

	// Create the TF-IDF model by:
	// splitting the doc into words and counting each instance of a word
	// calculating Inverse Document Frequency which says how important a word is
	tfidf := pkgTFIDF.NewTFIDF(corpus)

	//Calculate and store TF-IDF scores for each word
	tfidf.CalculateScores()

	// get the top 20 keywords
	tfidf.ExtractKeywords(20)
	return tfidf
}

// LoadCorpus loads the corpus from a text file and returns a slice of strings.
func LoadCorpus(filename string) ([]string, error) {
	var corpus []string
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		corpus = append(corpus, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return corpus, nil
}
