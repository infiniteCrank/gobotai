package tfidf

import (
	"math"
	"strings"
)

// TFIDF struct holds the term frequency and inverse document frequency.
type TFIDF struct {
	TermFrequency  map[string]float64 // Frequencies of terms in the corpus
	InverseDocFreq map[string]float64 // Inverse document frequencies for terms
}

// NewTFIDF creates a new TFIDF instance based on the provided corpus of documents.
func NewTFIDF(corpus []string) *TFIDF {
	tf := make(map[string]float64)  // Initialize map to store term frequencies
	idf := make(map[string]float64) // Initialize map to store inverse document frequencies

	// Calculate Term Frequency (TF)
	for _, doc := range corpus {
		words := strings.Fields(doc) // Split document into words
		for _, word := range words {
			tf[word]++ // Count occurrences of each word
		}
	}

	// Calculate Inverse Document Frequency (IDF)
	for term := range tf {
		idf[term] = math.Log(float64(len(corpus)) / (1 + float64(countDocumentsContainingTerm(corpus, term))))
	}

	// Return a new instance of TFIDF with calculated TF and IDF
	return &TFIDF{TermFrequency: tf, InverseDocFreq: idf}
}

// countDocumentsContainingTerm counts how many documents contain a specific term.
func countDocumentsContainingTerm(corpus []string, term string) int {
	count := 0 // Initialize count to zero
	for _, doc := range corpus {
		if strings.Contains(doc, term) { // Check if the term exists in the document
			count++ // Increment count if term is found
		}
	}
	return count // Return the total count
}
