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
			corpusFiles := []string{"go_textbook.md", "go_corpus.md"}
			tfidf := initializeTFIDF(corpusFiles)

			var inputQuery []string
			inputQuery = append(inputQuery, removeOutLiers)
			queryVec := pkgTFIDF.NewTFIDF(inputQuery)
			queryVec.CalculateScores()

			// format the data for training
			knnData := pkgKNN.CreateDataSet(tfidf.Corpus)

			// My query TF-IDF scores
			queryVecScores := queryVec.Scores

			// Retrieve the most relevant answer using KNN
			k := 21 // Number of neighbors to consider
			top := 1
			// get the top five headings
			answersCos := pkgKNN.KNNImproved(queryVecScores, knnData.Dataset, k, top)
			answersEucliedian := pkgKNN.KNN(queryVecScores, knnData.Dataset, k, top)
			answers := pkgKNN.KNNCombined(queryVecScores, knnData.Dataset, k, top, .60, .20)
			fmt.Printf("Most relevant content: %+v \n", answersCos)
			fmt.Printf("Most relevant content: %+v \n", answersEucliedian)
			response := "<b>Most relevant chapter: </b>" + strings.Join(answers, ", ") + "<br><br>\n"

			for _, corpusData := range knnData.FormattedCorpus {
				if contains(answers, corpusData.Answer) {
					for word := range queryVecScores {
						//check to see if the text contains a word from the original
						if strings.Contains(corpusData.Text, word) {
							formatedText := strings.ReplaceAll(corpusData.Text, "**", "<br>**")
							response += formatedText + "<br><br>\n"
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

func contains(slice []string, value string) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}

func initializeTFIDF(filenames []string) *pkgTFIDF.TFIDF {
	corpus, err := LoadCorpora(filenames)
	if err != nil {
		log.Fatal("Error loading corpora:", err)
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

func LoadCorpora(filenames []string) ([]string, error) {
	var combinedCorpus []string

	for _, filename := range filenames {
		corpus, err := LoadCorpus(filename)
		if err != nil {
			return nil, err // Return error if any file fails to load
		}
		combinedCorpus = append(combinedCorpus, corpus...) // Append the loaded corpus
	}

	return combinedCorpus, nil
}
