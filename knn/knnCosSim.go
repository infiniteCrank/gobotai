package knn

import (
	"math"
	"sort"
)

// CosineSimilarity calculates the cosine similarity between two TF-IDF vectors.
// Cosine similarity is particularly useful for sparse, high-dimensional data.
func CosineSimilarity(vec1, vec2 map[string]float64) float64 {
	var dot, norm1, norm2 float64
	for key, val := range vec1 {
		dot += val * vec2[key]
		norm1 += val * val
	}
	for _, val := range vec2 {
		norm2 += val * val
	}
	if norm1 == 0 || norm2 == 0 {
		return 0
	}
	return dot / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// Similarity represents the similarity between a data point and the query along with its index.
type Similarity struct {
	Index int     // Index of the DataPoint in the dataset
	Score float64 // Cosine similarity score (higher is more similar)
}

// KNNImproved finds the k nearest neighbors using cosine similarity and returns the top X answers based on weighted voting.
func KNNImproved(queryVec map[string]float64, dataset []DataPoint, k int, topX int) []string {
	// Calculate cosine similarity for each data point.
	similarities := make([]Similarity, len(dataset))
	for i, point := range dataset {
		score := CosineSimilarity(queryVec, point.Vector)
		similarities[i] = Similarity{Index: i, Score: score}
	}

	// Sort similarities in descending order (higher similarity is better).
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].Score > similarities[j].Score
	})

	// Use weighted voting based on similarity scores.
	answerWeights := make(map[string]float64)
	for i := 0; i < k && i < len(similarities); i++ {
		answer := dataset[similarities[i].Index].Answer
		// Weight the vote by the similarity score.
		answerWeights[answer] += similarities[i].Score
	}

	// Create a slice for sorting answers by their total weight.
	type AnswerWeight struct {
		Answer string
		Weight float64
	}
	var weightedAnswers []AnswerWeight
	for answer, weight := range answerWeights {
		weightedAnswers = append(weightedAnswers, AnswerWeight{Answer: answer, Weight: weight})
	}

	// Sort by descending weight.
	sort.Slice(weightedAnswers, func(i, j int) bool {
		return weightedAnswers[i].Weight > weightedAnswers[j].Weight
	})

	// Collect the top X answers.
	var topAnswers []string
	for i := 0; i < topX && i < len(weightedAnswers); i++ {
		topAnswers = append(topAnswers, weightedAnswers[i].Answer)
	}

	return topAnswers
}
