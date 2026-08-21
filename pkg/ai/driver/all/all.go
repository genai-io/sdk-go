// Package all registers every bundled protocol driver.
//
// It is for a program that does not know its models in advance — a CLI with a
// -model flag, a gateway that proxies whatever it is asked for. Those genuinely
// need every protocol linked in, and this saves them five blank imports:
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
//
// A program that does know should import the one or two drivers it uses.
// Reaching Gemini through this package costs 299 external packages where
// ai/driver/google alone costs 19, because this one links every vendor SDK
// including a cloud credential stack.
//
// The cost is not evenly spread, so it is worth knowing which import you are
// paying for. Non-standard-library packages each driver pulls in:
//
//	google                    19   plain HTTP; no vendor SDK
//	anthropic                 43
//	openai/chat               47
//	openai/responses          47
//	anthropic/vertex         271   Google's ADC credential stack, gRPC, protobuf
//
// A program that talks to one provider should import that driver alone. One
// that does not use Vertex AI should particularly leave anthropic/vertex out:
// it costs several times the other four together, and importing anthropic
// alone does not pull it in.
package all

import (
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic/vertex"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/google"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
)
