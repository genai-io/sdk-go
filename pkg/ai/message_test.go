package ai

import "testing"

// The signature rides with its thinking block, so it is found there and not
// in the readable text.
func TestThinkingSignatureIsTheBlocksOwn(t *testing.T) {
	c := Content{{Type: BlockText, Text: "answer"}, {Type: BlockThinking, Text: "hm", Signature: "sig"}}
	if got := c.ThinkingSignature(); got != "sig" {
		t.Errorf("ThinkingSignature = %q, want sig", got)
	}
	if got := TextContent("plain").ThinkingSignature(); got != "" {
		t.Errorf("no thinking gave %q", got)
	}
}
