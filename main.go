package main

import (
	"bufio"
	"fmt"
	"log"
	"os"

	pkgKNN "github.com/infiniteCrank/gobotai/knn"
	pkgTFIDF "github.com/infiniteCrank/gobotai/tfidf"
)

func main() {
	// init the corpus and supporting data
	tfidf := initializeTFIDF()
	var inputQuery []string
	inputQuery = append(inputQuery, "How do I define a function in go?")
	queryVec := pkgTFIDF.NewTFIDF(inputQuery)
	queryVec.CalculateScores()

	// 1. format the data for training
	knnData := pkgKNN.CreateDataSet(tfidf.Corpus)

	// My query TF-IDF scores
	queryVecScores := queryVec.Scores

	// Retrieve the most relevant answer using KNN
	k := 21 // Number of neighbors to consider
	answer := pkgKNN.KNN(queryVecScores, knnData.Dataset, k)

	// Print the result
	fmt.Println("Most relevant content:", answer)

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
