package main

import (
	"bufio"
	"fmt"
	"log"
	"net/http"
	"os"
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
	err := http.ListenAndServe(":8080", nil) // Start listening on port 8080
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
			// init the corpus and supporting data
			tfidf := initializeTFIDF()
			var inputQuery []string
			inputQuery = append(inputQuery, query)
			queryVec := pkgTFIDF.NewTFIDF(inputQuery)
			queryVec.CalculateScores()

			// 1. format the data for training
			knnData := pkgKNN.CreateDataSet(tfidf.Corpus)

			// My query TF-IDF scores
			queryVecScores := queryVec.Scores

			// Retrieve the most relevant answer using KNN
			k := 21 // Number of neighbors to consider
			answers := pkgKNN.KNN(queryVecScores, knnData.Dataset, k, 3)

			// Print the result
			fmt.Printf("Most relevant content: %+v \n", answers)
			response := "Most relevant content: " + strings.Join(answers, ", ") + "\n"
			corpusKeywords := tfidf.ExtractKeywords(20)
			var relatedKeywords []string
			for term := range corpusKeywords {
				if strings.Contains(strings.ToLower(inputQuery[0]), term) {
					relatedKeywords = append(relatedKeywords, term)
				}
			}
			fmt.Printf("Related Keywords: %+v \n", relatedKeywords)
			response += "It looks like you are looking for something related to " + strings.Join(relatedKeywords, ", ") + ".\n"
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
			response += "Here is the best headings to look under " + strings.Join(newAnswers, ", ") + " .\n"
			// Send the response back to the client
			err = conn.WriteJSON(map[string]string{"type": "response", "response": response})
			if err != nil {
				log.Println("Error on write:", err)
			}

		}
	}
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
