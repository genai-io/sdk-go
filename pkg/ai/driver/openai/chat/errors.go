package chat

import (
	"github.com/genai-io/sdk-go/pkg/ai/driver/internal/errs"
	"github.com/genai-io/sdk-go/pkg/ai/driver/openai/internal/oai"
)

// fail classifies this protocol's failures.
var fail = errs.For(name, oai.Details)
