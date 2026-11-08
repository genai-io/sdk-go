package ai

import (
	"context"
	"strings"
	"testing"
)

func BenchmarkAdmitRequest(b *testing.B) {
	for _, size := range []struct {
		name  string
		turns int
		image bool
	}{
		{"small", 1, false}, {"long", 200, false}, {"long_image_schema", 200, true},
	} {
		b.Run(size.name, func(b *testing.B) {
			req := &Request{ToolChoice: ToolChoiceNamed("search")}
			text := "hi"
			if size.turns > 1 {
				text = strings.Repeat("context ", 512)
			}
			for i := 0; i < size.turns; i++ {
				req.Messages = append(req.Messages, UserMessage(text))
			}
			if size.image {
				req.Messages[0].Content = append(req.Messages[0].Content, ImageBlock(Image{MediaType: "image/png", Data: strings.Repeat("YQ==", 256*1024)}))
				req.Tools = []Schema{ToolSchema[struct {
					Query string
					Limit int
				}]("search", "Search")}
				req.SamplingParams = map[string]any{"seed": int64(9007199254740993), "nested": map[string]any{"values": []any{1, "two", true}}}
			}
			ctx := context.Background()
			admit := func(context.Context, *Request) error { return nil }
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := admitRequest(ctx, req, admit); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
