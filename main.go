package main

import (
	"bufio"
	"fmt"
	"log"
	"os"

	"github.com/infiniteCrank/gobotai/tfidf"
)

func main() {
	// init the corpus and supporting data
	initialize()
}

func initialize() {
	// Load the existing corpus of training phrases as text
	corpus, err := LoadCorpus("go_corpus.md")
	if err != nil {
		log.Fatal("Error loading corpus:", err)
	}

	// Create the TF-IDF model by:
	// splitting the doc into words and counting each instance of a word
	// calculating Inverse Document Frequency which says how important a word is
	tfidf := tfidf.NewTFIDF(corpus)

	//Calculate and store TF-IDF vectors for each document
	vectors := tfidf.CalculateScores()

	// Store the vectors in the map
	for word, value := range vectors {
		fmt.Printf("Word: %+v Value: %+v \n", word, value)
	}

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
