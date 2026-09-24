package catalog

import (
	"strings"

	"github.com/genai-io/sdk-go/pkg/ai"
)

// ─── protocol behavior ───

var (
	// Current Claude takes adaptive thinking and rejects a non-default temperature.
	claudeAdaptiveNoTemp = ai.AnthropicCompat{ForceAdaptiveThinking: true, NoTemperature: true}
)

// ─── deployments ───

// vertexDeployment reads the project and region a Vertex-served model lives in
// from the row's own variables. The project has no default worth guessing, so
// a missing one is refused here rather than 400 later. Vertex takes
// Application Default Credentials, so no credential is among these.
func vertexDeployment(vars map[string]string, env func(string) string) (ai.ProtocolConfig, error) {
	project := strings.TrimSpace(env(vars["project"]))
	if project == "" {
		return nil, &MissingDeploymentError{
			EnvVars: []string{vars["project"]},
			Note: "Vertex needs a Google Cloud project. Credentials themselves come from " +
				"Application Default Credentials, not from a variable.",
		}
	}
	return ai.VertexConfig{Project: project, Region: strings.TrimSpace(env(vars["region"]))}, nil
}
