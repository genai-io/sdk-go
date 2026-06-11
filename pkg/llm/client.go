package llm

import (
	"context"
	"sync"
)

// defaultMaxTokens is the fallback max output tokens.
const defaultMaxTokens = 8192

// Client adapts a Provider to the LLM interface. It adds model selection,
// token tracking, and synchronous Complete() helper on top of any Provider.
type Client struct {
	mu             sync.RWMutex
	provider       Provider
	model          string
	maxTokens      int
	thinkingEffort string
}

// NewClient wraps a Provider with model selection and token tracking.
// maxTokens=0 means resolve from provider metadata or fall back to defaultMaxTokens.
func NewClient(p Provider, model string, maxTokens int) *Client {
	return &Client{provider: p, model: model, maxTokens: maxTokens}
}

// Infer implements LLM — streaming inference.
func (c *Client) Infer(ctx context.Context, req InferRequest) (<-chan Chunk, error) {
	c.mu.RLock()
	p := c.provider
	model := c.model
	maxTokens := c.maxTokens
	thinking := c.thinkingEffort
	c.mu.RUnlock()

	opts := CompletionOptions{
		Model:          model,
		Messages:       req.Messages,
		Tools:          req.Tools,
		SystemPrompt:   req.System,
		MaxTokens:      resolveMaxTokens(maxTokens, p, model),
		ThinkingEffort: thinking,
	}

	srcCh := p.Stream(ctx, opts)

	ch := make(chan Chunk, 8)
	go func() {
		defer close(ch)
		send := func(chunk Chunk) bool {
			select {
			case ch <- chunk:
				return true
			case <-ctx.Done():
				return false
			}
		}
		for sc := range srcCh {
			switch sc.Type {
			case ChunkTypeText:
				if !send(Chunk{Text: sc.Text}) {
					return
				}
			case ChunkTypeThinking:
				if !send(Chunk{Thinking: sc.Text}) {
					return
				}
			case ChunkTypeDone:
				if !send(Chunk{Done: true, Response: toInferResponse(sc.Response)}) {
					return
				}
			case ChunkTypeError:
				send(Chunk{Err: sc.Error})
				return
			}
		}
	}()

	return ch, nil
}

// Complete runs inference and collects all chunks into a single response.
// Convenience method for non-streaming use cases.
func (c *Client) Complete(ctx context.Context, req InferRequest) (*InferResponse, error) {
	ch, err := c.Infer(ctx, req)
	if err != nil {
		return nil, err
	}
	var response InferResponse
	gotDone := false
	for chunk := range ch {
		if chunk.Err != nil {
			return nil, chunk.Err
		}
		if chunk.Done && chunk.Response != nil {
			response = *chunk.Response
			gotDone = true
		}
	}
	if !gotDone {
		return nil, context.DeadlineExceeded
	}
	return &response, nil
}

// InputLimit returns the model's max input token capacity.
func (c *Client) InputLimit() int {
	c.mu.RLock()
	p := c.provider
	model := c.model
	c.mu.RUnlock()
	return inputLimitFromProvider(p, model)
}

// SetThinkingEffort changes the thinking/reasoning effort value.
func (c *Client) SetThinkingEffort(effort string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.thinkingEffort = effort
}

// ThinkingEffort returns the current thinking/reasoning effort value.
func (c *Client) ThinkingEffort() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.thinkingEffort
}

// Name returns the provider name.
func (c *Client) Name() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.provider == nil {
		return ""
	}
	return c.provider.Name()
}

// ModelID returns the model identifier.
func (c *Client) ModelID() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.model
}

// Provider returns the underlying Provider.
func (c *Client) Provider() Provider {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.provider
}

func resolveMaxTokens(maxTokens int, p Provider, model string) int {
	if maxTokens > 0 {
		return maxTokens
	}
	if limit := outputLimitFromProvider(p, model); limit > 0 {
		return limit
	}
	return defaultMaxTokens
}

func inputLimitFromProvider(p Provider, model string) int {
	if p == nil {
		return 0
	}
	models, err := p.ListModels(context.TODO())
	if err == nil {
		for _, m := range models {
			if m.ID == model && m.InputTokenLimit > 0 {
				return m.InputTokenLimit
			}
		}
	}
	return 0
}

func outputLimitFromProvider(p Provider, model string) int {
	if p == nil {
		return 0
	}
	models, err := p.ListModels(context.TODO())
	if err == nil {
		for _, m := range models {
			if m.ID == model && m.OutputTokenLimit > 0 {
				return m.OutputTokenLimit
			}
		}
	}
	return 0
}

func toInferResponse(r *CompletionResponse) *InferResponse {
	if r == nil {
		return nil
	}
	return &InferResponse{
		Content:           r.Content,
		Thinking:          r.Thinking,
		ThinkingSignature: r.ThinkingSignature,
		ToolCalls:         r.ToolCalls,
		StopReason:        StopReason(r.StopReason),
		TokensIn:          r.Usage.InputTokens,
		TokensOut:         r.Usage.OutputTokens,
		CacheCreateTokens: r.Usage.CacheCreationInputTokens,
		CacheReadTokens:   r.Usage.CacheReadInputTokens,
	}
}
