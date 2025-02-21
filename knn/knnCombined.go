package knn

import "sort"

// KNNCombined computes a combined similarity score using both cosine similarity and a normalized Euclidean similarity.
// The Euclidean similarity is calculated as 1/(1 + EuclideanDistance), so lower distance produces higher similarity.
// wCosine and wEuclid are weights for tuning the influence of each metric.
func KNNCombined(queryVec map[string]float64, dataset []DataPoint, k int, topX int, wCosine, wEuclid float64) []string {
	// CombinedScore holds the index of the data point and its computed combined score.
	type CombinedScore struct {
		Index int
		Score float64
	}

	combinedScores := make([]CombinedScore, len(dataset))
	for i, point := range dataset {
		// Compute Euclidean distance and convert to similarity.
		euclidDistance := EuclideanDistance(queryVec, point.Vector)
		euclidSim := 1.0 / (1.0 + euclidDistance) // Normalized: lower distance -> higher similarity.

		// Compute cosine similarity.
		cosineSim := CosineSimilarity(queryVec, point.Vector)

		// Combine the two metrics using a weighted sum.
		combinedScore := wCosine*cosineSim + wEuclid*euclidSim

		combinedScores[i] = CombinedScore{Index: i, Score: combinedScore}
	}

	// Sort the data points by descending combined score.
	sort.Slice(combinedScores, func(i, j int) bool {
		return combinedScores[i].Score > combinedScores[j].Score
	})

	// Use weighted voting among the top k neighbors.
	answerWeights := make(map[string]float64)
	for i := 0; i < k && i < len(combinedScores); i++ {
		answer := dataset[combinedScores[i].Index].Answer
		answerWeights[answer] += combinedScores[i].Score
	}

	// Sort the answers by total weight.
	type AnswerWeight struct {
		Answer string
		Weight float64
	}
	var weightedAnswers []AnswerWeight
	for answer, weight := range answerWeights {
		weightedAnswers = append(weightedAnswers, AnswerWeight{Answer: answer, Weight: weight})
	}
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
