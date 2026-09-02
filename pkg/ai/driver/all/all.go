// Package all registers every bundled protocol driver.
//
//	import _ "github.com/genai-io/sdk-go/pkg/ai/driver/all"
//
// It costs every protocol's dependencies, and they are not the same size.
// google needs none at all — it speaks plain HTTP. anthropic and the two openai
// packages each pull in one vendor SDK. anthropic/vertex pulls in Google's
// Application Default Credentials stack, and with it gRPC and protobuf, which
// dwarfs the rest put together. A binary that talks to one provider should
// blank-import that provider's package instead; run "go mod graph" against your
// own build for the current figures.
package all

import (
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/anthropic/vertex"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/google"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/chat"
	_ "github.com/genai-io/sdk-go/pkg/ai/driver/openai/responses"
)
