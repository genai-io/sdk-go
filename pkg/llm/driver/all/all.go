// Package all registers every bundled protocol driver.
//
// Import it for its side effect when you want llm.Open to handle any model the
// catalog can name:
//
//	import _ "github.com/genai-io/sdk-go/pkg/llm/driver/all"
//
// The cost is not evenly spread, so it is worth knowing which import you are
// paying for. Non-standard-library packages each driver pulls in:
//
//	google              2    plain HTTP; no vendor SDK
//	anthropic          14
//	openaichat         16
//	openairesp         16
//	anthropicvertex   130    Google's ADC credential stack, gRPC and protobuf
//
// Importing this package costs the union, 144. A program that talks to one
// provider should import that driver alone and leave the rest out; a program
// that does not use Vertex AI should particularly leave anthropicvertex out,
// since it accounts for nearly all of the weight.
package all

import (
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/anthropic"
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/anthropicvertex"
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/google"
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/openaichat"
	_ "github.com/genai-io/sdk-go/pkg/llm/driver/openairesp"
)
