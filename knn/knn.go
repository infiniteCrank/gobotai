package knn

import (
	"math"
	"sort"
	"strings"

	pkgTFIDF "github.com/infiniteCrank/gobotai/tfidf"
)

// DataPoint represents a single entry in the dataset for KNN.
// It contains a vector (representing its TF-IDF values), the response associated with that entry, and the associated intent.
type DataPoint struct {
	Vector map[string]float64 // TF-IDF vector for the data point
	Answer string             // The response associated with this data point
	Intent string             // The identified intent of the data point (optional)
}
type DataEntry struct {
	Text   string
	Answer string
}
type KNNData struct {
	Dataset         []DataPoint
	FormattedCorpus []DataEntry
}

// EuclideanDistance calculates the Euclidean distance between two vectors.
func EuclideanDistance(vec1, vec2 map[string]float64) float64 {
	var sum float64
	// Iterate over all keys in vec1 to compute the distance
	for key := range vec1 {
		// If key exists in vec2, compute the squared difference, otherwise treat vec2[key] as 0
		diff := vec1[key] - vec2[key]
		sum += diff * diff
	}
	// Include terms that are in vec2 but not in vec1
	for key := range vec2 {
		if _, exists := vec1[key]; !exists {
			sum += vec2[key] * vec2[key]
		}
	}
	return math.Sqrt(sum) // Return the square root of the sum of squared differences
}

// Distance represents the distance between a data point and a query along with its index in the dataset.
type Distance struct {
	Index int     // Index of the original DataPoint in the dataset
	Value float64 // Calculated distance to the query point
}

// ByDistance is a type that implements sorting of Distance slices based on their Value.
type ByDistance []Distance

// Len returns the number of elements in the collection.
func (a ByDistance) Len() int {
	return len(a)
}

// Swap exchanges the elements with indexes i and j.
func (a ByDistance) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

// Less reports whether the element with index i should sort before the element with index j.
func (a ByDistance) Less(i, j int) bool {
	return a[i].Value < a[j].Value // Sort by increasing distance
}

// KNN function finds the k nearest neighbors to a given query vector.
// It returns the top X most common answers among the nearest neighbors.
func KNN(queryVec map[string]float64, dataset []DataPoint, k int, topX int) []string {
	distances := make([]Distance, len(dataset)) // Initialize distances slice

	// Calculate the Euclidean distance for each data point in the dataset
	for i, point := range dataset {
		dist := EuclideanDistance(queryVec, point.Vector) // Calculate distance
		distances[i] = Distance{Index: i, Value: dist}    // Store index and distance
	}

	// Sort distances to find the nearest neighbors
	sort.Sort(ByDistance(distances))

	// Count the frequency of answers among the k nearest neighbors
	answerCount := make(map[string]int)
	for i := 0; i < k && i < len(distances); i++ {
		answerCount[dataset[distances[i].Index].Answer]++
	}

	// Create a slice to hold answers and their counts
	type AnswerFrequency struct {
		Answer string
		Count  int
	}
	var frequencies []AnswerFrequency

	// Fill the frequencies slice
	for answer, count := range answerCount {
		frequencies = append(frequencies, AnswerFrequency{Answer: answer, Count: count})
	}

	// Sort frequencies by count in descending order
	sort.Slice(frequencies, func(i, j int) bool {
		return frequencies[i].Count > frequencies[j].Count
	})

	// Collect the top X answers
	var topAnswers []string
	for i := 0; i < topX && i < len(frequencies); i++ {
		topAnswers = append(topAnswers, frequencies[i].Answer)
	}

	return topAnswers // Return the top X most common answers among the nearest neighbors
}

func FormatCorpus(corpus string) KNNData {
	var knnData KNNData

	// Trim the entire corpus to remove any leading/trailing whitespace.
	corpus = strings.TrimSpace(corpus)

	// Split the document by the section delimiter "## |||"
	sections := strings.Split(corpus, "## |||")

	for _, section := range sections {
		section = strings.TrimSpace(section)
		if section == "" {
			continue
		}

		// Split each section into chapters by the delimiter "###"
		chapters := strings.Split(section, "###")
		var currentAnswer string

		for idx, chapter := range chapters {
			chapter = strings.TrimSpace(chapter)
			if chapter == "" {
				continue
			}

			// Create a new DataEntry for each chapter to avoid reusing the same variable.
			var entry DataEntry

			if idx == 0 {
				// For the first chapter, split further by "|||"
				answerText := strings.Split(chapter, "|||")
				if len(answerText) > 0 {
					entry.Answer = strings.TrimSpace(answerText[0])
					currentAnswer = entry.Answer // Save the answer for subsequent chapters.
				} else {
					entry.Answer = ""
					currentAnswer = ""
				}
				if len(answerText) > 1 {
					entry.Text = strings.TrimSpace(answerText[1])
				} else {
					entry.Text = "no text available"
				}
			} else {
				// For subsequent chapters, reuse the saved answer.
				entry.Answer = currentAnswer
				entry.Text = chapter
			}

			knnData.FormattedCorpus = append(knnData.FormattedCorpus, entry)
		}
	}

	return knnData
}

// CreateDataSet converts the formatted corpus into a dataset of DataPoints for KNN.
func CreateDataSet(corpus string) KNNData {
	knnData := FormatCorpus(corpus)
	var dataPoint DataPoint
	for _, corpusData := range knnData.FormattedCorpus {
		// Create a TF-IDF vector for each text entry.
		tfidf := pkgTFIDF.NewTFIDF([]string{corpusData.Text})
		tfidf.CalculateScores()
		dataPoint.Vector = tfidf.Scores
		dataPoint.Answer = corpusData.Answer
		knnData.Dataset = append(knnData.Dataset, dataPoint)
	}
	return knnData
}
