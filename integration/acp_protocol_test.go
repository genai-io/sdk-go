package integration_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/genai-io/sdk-go/pkg/acp"
	"github.com/genai-io/sdk-go/pkg/acp/protocol"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
)

type acpCapture struct {
	Direction string `json:"direction"`
	Line      string `json:"line"`
}

func fixtureValue[T any](data json.RawMessage) (T, error) {
	var value T
	err := json.Unmarshal(data, &value)
	return value, err
}

func sameJSON(left, right []byte) bool {
	decode := func(data []byte) (any, error) {
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.UseNumber()
		var value any
		err := decoder.Decode(&value)
		return value, err
	}
	a, errA := decode(left)
	b, errB := decode(right)
	return errA == nil && errB == nil && reflect.DeepEqual(a, b)
}

// Replay actual official-client bytes through the Go connection. Callbacks are
// fixture behavior, not a session admission or cancellation implementation.
func TestACPTypesAgainstOfficialClientFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/acp/baseline.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	if !scanner.Scan() {
		t.Fatal("missing capture provenance")
	}
	var metadata struct {
		SDK                string `json:"sdk"`
		Version            string `json:"version"`
		Commit             string `json:"commit"`
		TargetSchemaCommit string `json:"targetSchemaCommit"`
	}
	if err := json.Unmarshal(scanner.Bytes(), &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.SDK != "@agentclientprotocol/sdk" || metadata.Version != "2.0.2" ||
		metadata.Commit != "f2f218d314ee8f42c37c466b55d2b917ddc0006c" ||
		metadata.TargetSchemaCommit != "e2d83349bc9e28a4059ea3acefa1c91c966f74e2" {
		t.Fatalf("unexpected capture provenance: %+v", metadata)
	}

	var frames []acpCapture
	results := make(map[string]json.RawMessage)
	permissionParams := make(map[string]json.RawMessage)
	var update json.RawMessage
	for scanner.Scan() {
		var frame acpCapture
		if err := json.Unmarshal(scanner.Bytes(), &frame); err != nil {
			t.Fatal(err)
		}
		var message struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
			Result json.RawMessage `json:"result"`
		}
		if err := json.Unmarshal([]byte(frame.Line), &message); err != nil {
			t.Fatal(err)
		}
		if len(message.Result) != 0 {
			results[string(message.ID)] = message.Result
		}
		if message.Method == "session/request_permission" {
			permissionParams[string(message.ID)] = message.Params
		}
		if message.Method == "session/update" {
			update = message.Params
		}
		frames = append(frames, frame)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(frames) != 14 || len(permissionParams) != 2 || update == nil {
		t.Fatal("incomplete official client capture")
	}

	// Include cold schema compilation under the race detector in the deadline.
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(cancel)
	var turns atomic.Int32
	var cancelled atomic.Bool
	conn, peer := acpPipeContext(t, ctx, acp.Handlers{
		Request: func(ctx context.Context, c *acp.Conn, method string, params json.RawMessage) (any, error) {
			switch method {
			case "initialize":
				if _, err := fixtureValue[protocol.InitializeRequest](params); err != nil {
					return nil, err
				}
				return fixtureValue[protocol.InitializeResponse](results["0"])
			case "session/new":
				if _, err := fixtureValue[protocol.NewSessionRequest](params); err != nil {
					return nil, err
				}
				return fixtureValue[protocol.NewSessionResponse](results["1"])
			case "session/prompt":
				if _, err := fixtureValue[protocol.PromptRequest](params); err != nil {
					return nil, err
				}
				turn := turns.Add(1)
				permissionID := fmt.Sprintf(`"permission-%d"`, turn)
				request, err := fixtureValue[protocol.RequestPermissionRequest](permissionParams[permissionID])
				if err != nil {
					return nil, err
				}
				decision, err := c.RequestPermission(ctx, request)
				if err != nil {
					return nil, err
				}
				wire, err := json.Marshal(decision)
				if err != nil || !sameJSON(wire, results[permissionID]) {
					return nil, errors.New("permission result differs from official client capture")
				}
				if turn == 1 {
					notification, err := fixtureValue[protocol.SessionNotification](update)
					if err != nil {
						return nil, err
					}
					if err := c.SessionUpdate(ctx, notification); err != nil {
						return nil, err
					}
				} else if !cancelled.Load() {
					return nil, errors.New("permission response overtook cancellation notification")
				}
				return fixtureValue[protocol.PromptResponse](results[fmt.Sprint(turn+1)])
			default:
				return nil, &acp.Error{Code: -32601, Message: "Method not found"}
			}
		},
		Notification: func(_ context.Context, _ *acp.Conn, method string, params json.RawMessage) error {
			if method != "session/cancel" {
				return errors.New("unexpected fixture notification")
			}
			if _, err := fixtureValue[protocol.CancelNotification](params); err != nil {
				return err
			}
			cancelled.Store(true)
			return nil
		},
	})

	// Outgoing Go request IDs need not match the official fixture's mock agent.
	ids := make(map[string]json.RawMessage)
	for i, frame := range frames {
		var expected map[string]json.RawMessage
		if err := json.Unmarshal([]byte(frame.Line), &expected); err != nil {
			t.Fatal(err)
		}
		switch frame.Direction {
		case "client_to_agent":
			if replacement, ok := ids[string(expected["id"])]; ok {
				expected["id"] = replacement
			}
			wire, err := json.Marshal(expected)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := peer.input.Write(append(wire, '\n')); err != nil {
				t.Fatal(err)
			}
		case "agent_to_client":
			message := peer.read(t)
			wire, err := jsonrpc.EncodeMessage(message)
			if err != nil {
				t.Fatal(err)
			}
			var actual map[string]json.RawMessage
			if err := json.Unmarshal(wire, &actual); err != nil {
				t.Fatal(err)
			}
			if len(expected["method"]) != 0 && len(expected["id"]) != 0 {
				if len(actual["id"]) == 0 {
					t.Fatal("permission request lost its ID")
				}
				ids[string(expected["id"])] = actual["id"]
				expected["id"] = actual["id"]
			}
			want, err := json.Marshal(expected)
			if err != nil {
				t.Fatal(err)
			}
			if !sameJSON(wire, want) {
				t.Fatalf("frame %d = %s, want %s", i, wire, want)
			}
		default:
			t.Fatalf("unknown capture direction %q", frame.Direction)
		}
	}
	if turns.Load() != 2 || !cancelled.Load() || conn.Err() != nil {
		t.Fatalf("fixture turns=%d, cancelled=%t, connection=%v", turns.Load(), cancelled.Load(), conn.Err())
	}
}
