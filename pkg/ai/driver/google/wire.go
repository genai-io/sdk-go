package google

// The Gemini generateContent wire format.

type content struct {
	Role  string  `json:"role,omitempty"`
	Parts []*part `json:"parts,omitempty"`
}

type part struct {
	Text             string            `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature []byte            `json:"thoughtSignature,omitempty"`
	InlineData       *blob             `json:"inlineData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
}

type blob struct {
	MIMEType string `json:"mimeType,omitempty"`
	// Data is bytes, not the base64 text they arrive as: encoding/json writes
	// a []byte as base64, which is the shape the field wants.
	Data []byte `json:"data,omitempty"`
}

type functionCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

type functionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name,omitempty"`
	Response map[string]any `json:"response,omitempty"`
}

// generateRequest is the body of :generateContent and :streamGenerateContent.
type generateRequest struct {
	Contents          []*content        `json:"contents,omitempty"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	Tools             []*tool           `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
}

type tool struct {
	FunctionDeclarations []*functionDeclaration `json:"functionDeclarations,omitempty"`
}

type functionDeclaration struct {
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	// ParametersJSONSchema takes the schema as written. The neighbouring
	// "parameters" field takes Gemini's own Schema type, which is a lossy
	// translation of the same thing.
	ParametersJSONSchema any `json:"parametersJsonSchema,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig *functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type functionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// Function-calling modes, as the API spells them.
const (
	modeAny  = "ANY"
	modeNone = "NONE"
)

type generationConfig struct {
	MaxOutputTokens    int32           `json:"maxOutputTokens,omitempty"`
	Temperature        *float32        `json:"temperature,omitempty"`
	StopSequences      []string        `json:"stopSequences,omitempty"`
	ThinkingConfig     *thinkingConfig `json:"thinkingConfig,omitempty"`
	ResponseMIMEType   string          `json:"responseMimeType,omitempty"`
	ResponseJSONSchema any             `json:"responseJsonSchema,omitempty"`
}

type thinkingConfig struct {
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
	// ThinkingBudget is a pointer so a deliberate zero — thinking off — is
	// sent rather than omitted.
	ThinkingBudget *int32 `json:"thinkingBudget,omitempty"`
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
}

type generateResponse struct {
	Candidates    []*candidate   `json:"candidates,omitempty"`
	ResponseID    string         `json:"responseId,omitempty"`
	UsageMetadata *usageMetadata `json:"usageMetadata,omitempty"`
}

type candidate struct {
	Content      *content `json:"content,omitempty"`
	FinishReason string   `json:"finishReason,omitempty"`
}

type usageMetadata struct {
	PromptTokenCount     int32 `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount int32 `json:"candidatesTokenCount,omitempty"`
	// ThoughtsTokenCount is billed at the output rate but is not inside
	// candidatesTokenCount, so a thinking turn is under-reported without it.
	ThoughtsTokenCount      int32 `json:"thoughtsTokenCount,omitempty"`
	CachedContentTokenCount int32 `json:"cachedContentTokenCount,omitempty"`
}

// countTokensRequest carries the whole request rather than bare contents: the
// system instruction and the tool declarations count against the window too,
// and the wrapper is the only form that accepts them.
type countTokensRequest struct {
	GenerateContentRequest *countTokensInner `json:"generateContentRequest,omitempty"`
}

type countTokensInner struct {
	// Model is required inside the wrapper, in "models/{id}" form.
	Model             string     `json:"model"`
	Contents          []*content `json:"contents,omitempty"`
	SystemInstruction *content   `json:"systemInstruction,omitempty"`
	Tools             []*tool    `json:"tools,omitempty"`
}

type countTokensResponse struct {
	TotalTokens int32 `json:"totalTokens,omitempty"`
}

type modelList struct {
	Models        []listedModel `json:"models,omitempty"`
	NextPageToken string        `json:"nextPageToken,omitempty"`
}

type listedModel struct {
	Name             string `json:"name,omitempty"`
	DisplayName      string `json:"displayName,omitempty"`
	InputTokenLimit  int32  `json:"inputTokenLimit,omitempty"`
	OutputTokenLimit int32  `json:"outputTokenLimit,omitempty"`
	// SupportedGenerationMethods is what the model can actually be asked to do.
	// The listing carries embedding, TTS, image and live models alongside the
	// ones that answer a prompt, and only this tells them apart.
	SupportedGenerationMethods []string `json:"supportedGenerationMethods,omitempty"`
}

// apiError is the Google API error envelope: {"error":{code,message,status}}.
type apiError struct {
	Error struct {
		Code    int    `json:"code,omitempty"`
		Message string `json:"message,omitempty"`
		Status  string `json:"status,omitempty"`
	} `json:"error"`
}
