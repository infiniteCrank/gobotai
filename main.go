package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"

	"github.com/gorilla/websocket"
	pkgKNN "github.com/infiniteCrank/gobotai/knn"
	pkgTFIDF "github.com/infiniteCrank/gobotai/tfidf"
)

func main() {

	server := newServer()

	// Handle WebSocket connections
	http.HandleFunc("/ws", server.handleWebSocket)

	// Serve static files
	http.Handle("/", http.FileServer(http.Dir("./frontend")))

	log.Println("Server started on :8080")
	err := http.ListenAndServe("127.0.0.1:8080", nil) // Start listening on port 8080
	if err != nil {
		log.Fatal("ListenAndServe: ", err) // Log any errors starting the server
	}
}

// Server holds all lobbies.
type Server struct {
	upgrader websocket.Upgrader
}

// Initialize a new server.
func newServer() *Server {
	return &Server{
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				//return r.Header.Get("Origin") == "https://sectorj.com"
				return true // Allow all connections for simplicity
			},
		},
	}
}

// Handle the websocket interaction for user queries
func (s Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	// upgrader for web sockets
	var upgrader = websocket.Upgrader{}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Error during connection upgrade:", err)
		return
	}
	defer conn.Close()

	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			log.Println("Error on read:", err)
			break
		}

		switch msg["type"] {
		case "query":
			query := msg["query"].(string)

			// Compile the regular expression to match "go" or "golang"
			re := regexp.MustCompile(`(?i)\b(go|golang)\b`)
			re2 := regexp.MustCompile(`(?i)\b(Go|GoLang)\b`)

			// Remove the matched words from the string
			removeOutLiers := re.ReplaceAllString(query, "")
			removeOutLiers = re2.ReplaceAllString(removeOutLiers, "")

			// init the corpus and supporting data
			tfidf := initializeTFIDF()
			var inputQuery []string
			inputQuery = append(inputQuery, removeOutLiers)
			queryVec := pkgTFIDF.NewTFIDF(inputQuery)
			queryVec.CalculateScores()

			// format the data for training
			knnData := pkgKNN.CreateDataSet(tfidf.Corpus)

			// My query TF-IDF scores
			queryVecScores := queryVec.Scores
			// match scores with original corpus words for accuracy
			queryVecScores = fixScores(queryVecScores, tfidf.Scores)

			// Retrieve the most relevant answer using KNN
			k := 21 // Number of neighbors to consider
			top := 5
			// get the top five headings
			answers := pkgKNN.KNNImproved(queryVecScores, knnData.Dataset, k, top)
			fmt.Printf("Most relevant content: %+v \n", answers)
			//response := "<b>Most relevant content: </b>" + strings.Join(answers, ", ") + "<br><br>\n"

			// extract the top 20 keywords from the query
			corpusKeywords := tfidf.ExtractKeywords(20)

			// search the user query query for the top 20 keywords
			var relatedKeywords []string
			for term := range corpusKeywords {
				if strings.Contains(strings.ToLower(inputQuery[0]), term) {
					relatedKeywords = append(relatedKeywords, term)
				}
			}
			fmt.Printf("Related Keywords: %+v \n", relatedKeywords)
			//response += "It looks like you are looking for something related to " + strings.Join(relatedKeywords, ", ") + ".<br><br>\n"

			// create a new query from related keywords that should be more accurate than user query
			var newQuery string
			for _, keyword := range relatedKeywords {
				newQuery += " " + keyword
			}
			var stringSlice []string
			stringSlice = append(stringSlice, newQuery)

			//run the new keyword query
			newQueryVec := pkgTFIDF.NewTFIDF(stringSlice)
			newQueryVec.CalculateScores()
			newQueryVecScores := newQueryVec.Scores

			// match scores with original corpus words for accuracy
			newQueryVecScores = fixScores(newQueryVecScores, tfidf.Scores)
			// get three new topics from keywords
			newAnswers := pkgKNN.KNNImproved(newQueryVecScores, knnData.Dataset, k, 3)
			fmt.Printf("Best content: %+v \n", newAnswers)
			//response += "Here is the best headings to look under " + strings.Join(newAnswers, ", ") + " .<br><br>\n"

			// add query words that are greater than 4 characters to the answers list
			for word := range queryVecScores {
				if len(word) > 4 {
					newAnswers = append(newAnswers, word)
					answers = append(answers, word)
				}
			}

			response := ""
			for _, corpusData := range knnData.FormattedCorpus {

				//check to see if this part of the corpus is releveant to the AI answers
				if contains(newAnswers, corpusData.Answer) {

					for _, keyword := range relatedKeywords {
						//check to see if the text contains a key word
						if strings.Contains(corpusData.Text, keyword) {
							response += corpusData.Text + "<br><br>\n"
							break
						}
					}
					//check to see if this part of the corpus is related to original answers
				} else if contains(answers, corpusData.Answer) {
					for word := range queryVecScores {
						//check to see if the text contains a word from the original
						if strings.Contains(corpusData.Text, word) {
							response += corpusData.Text + "<br><br>\n"
							break
						}
					}
				}
			}

			// Send the response back to the client
			err = conn.WriteJSON(map[string]string{"type": "response", "response": response})
			if err != nil {
				log.Println("Error on write:", err)
			}

			//create word maps
			scoresJson, err := json.Marshal(tfidf.Scores)
			if err != nil {
				log.Printf("trouble in paradise: %+v", err)
			}
			queryScores, err := json.Marshal(queryVecScores)
			if err != nil {
				log.Printf("trouble in paradise: %+v", err)
			}
			err = conn.WriteJSON(map[string]string{"type": "scores", "corpus": string(scoresJson), "query": string(queryScores)})
			if err != nil {
				log.Println("Error on write:", err)
			}

		}
	}
}

func fixScores(queryVecScores map[string]float64, corpusVecScores map[string]float64) map[string]float64 {
	for queryWord, queryScore := range queryVecScores {
		for corpusWord, corpusScore := range corpusVecScores {
			if queryWord == corpusWord {
				fmt.Printf("changing %+v score from %+v to %+v \n", queryWord, queryScore, corpusScore)
				queryVecScores[queryWord] = corpusScore
			}
		}
	}
	return queryVecScores
}

func contains(slice []string, value string) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}

func initializeTFIDF() *pkgTFIDF.TFIDF {
	// Load the existing corpus of training phrases as text
	corpus, err := LoadCorpus("go_textbook.md")
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
