package llm

import (
	"context"
	"fmt"
)

// Complete sends a one-shot completion through a Provider and collects the
// full response. For repeated calls with model selection, use NewClient instead.
func Complete(ctx context.Context, p Provider, opts CompletionOptions) (CompletionResponse, error) {
	var response CompletionResponse

	streamChan := p.Stream(ctx, opts)

	gotDone := false
	for chunk := range streamChan {
		switch chunk.Type {
		case ChunkTypeText:
			response.Content += chunk.Text
		case ChunkTypeDone:
			if chunk.Response != nil {
				return *chunk.Response, nil
			}
			gotDone = true
		case ChunkTypeError:
			return response, chunk.Error
		}
	}

	if !gotDone {
		return response, fmt.Errorf("stream closed without completion")
	}
	return response, nil
}
