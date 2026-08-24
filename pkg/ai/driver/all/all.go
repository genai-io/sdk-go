// Package all registers every bundled protocol driver.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
//
//	google                    19   plain HTTP; no vendor SDK
//	anthropic                 43
//	openai/chat               47
//	openai/responses          47
//	anthropic/vertex         271   Google's ADC credential stack, gRPC, protobuf
package all

import (
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic/vertex"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/google"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
)
