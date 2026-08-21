// Command gemini talks to Google's Gemini models, with an image.
//
// This driver speaks the REST API directly rather than through
// google.golang.org/genai — that SDK also serves Vertex AI, so it carries gRPC,
// protobuf and Google's cloud credential stack. Reaching Gemini with an API key
// costs 19 external packages here instead of around 190.
//
//	export GEMINI_API_KEY=...
//	go run ./examples/gemini path/to/picture.png
package main

import (
	"context"
	"encoding/base64"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	"github.com/genai-io/sdk-go/pkg/ai"
	"github.com/genai-io/sdk-go/pkg/ai/auth"

	_ "github.com/genai-io/sdk-go/pkg/ai/driver/google"
)

func main() {
	model := flag.String("model", "google/gemini-3-pro", "model reference")
	flag.Parse()

	if err := run(*model, flag.Arg(0)); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(ref, imagePath string) error {
	client, err := auth.Client(ref)
	if err != nil {
		return err
	}

	message := ai.UserMessage("What is in this picture? Answer in one sentence.")
	if imagePath != "" {
		image, err := loadImage(imagePath)
		if err != nil {
			return err
		}
		// A model that does not accept images refuses this before the request
		// leaves, naming the model — rather than dropping the picture and
		// answering about something it never saw.
		if !client.Model().Accepts(ai.ModalityImage) {
			return fmt.Errorf("%s does not accept images", client.Model())
		}
		message = ai.UserMessage("What is in this picture? Answer in one sentence.", image)
	}

	// Gemini 3 takes a thinking *level* where 2.5 took a token budget. Both are
	// the same rung here; the catalog carries which one this model wants.
	resp, err := client.Complete(context.Background(),
		[]ai.Message{message}, ai.WithEffort(ai.EffortLow))
	if err != nil {
		return err
	}

	fmt.Println(resp.Text())
	fmt.Printf("\n\033[2m%d in / %d out\033[0m\n", resp.Usage.TotalInput(), resp.Usage.Output)
	return nil
}

// loadImage reads a file into the inline form the SDK carries: base64 text and
// its media type, with no data: URI prefix — each driver adds whatever framing
// its own protocol wants.
func loadImage(path string) (ai.Image, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return ai.Image{}, err
	}
	mediaType := http.DetectContentType(raw)
	return ai.Image{
		MediaType: mediaType,
		Data:      base64.StdEncoding.EncodeToString(raw),
		FileName:  filepath.Base(path),
	}, nil
}
