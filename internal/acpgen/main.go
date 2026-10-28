// Command acpgen generates the pinned stable ACP wire model.
package main

import (
	"bytes"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/format"
	"maps"
	"os"
	"slices"
	"strings"
	"unicode"
)

type node map[string]any

var defs map[string]node
var out bytes.Buffer
var roots = strings.Fields("InitializeRequest InitializeResponse NewSessionRequest NewSessionResponse PromptRequest PromptResponse CancelNotification SessionNotification RequestPermissionRequest RequestPermissionResponse")

var sourcePin = struct {
	WireVersion   int
	SchemaVersion string
	Commit        string
	SchemaBlob    string
	MetadataBlob  string
}{
	WireVersion:   1,
	SchemaVersion: "schema-v1.26.0",
	Commit:        "e2d83349bc9e28a4059ea3acefa1c91c966f74e2",
	SchemaBlob:    "d6f7731bf74b3dbe3b9bbda52957a7637a12e688",
	MetadataBlob:  "b9c67caa417dd2285a31d07309a215a6110bbfaf",
}

func asObject(v any) node {
	if v == nil {
		return node{}
	}
	return node(v.(map[string]any))
}
func asArray(v any) []any {
	if v == nil {
		return nil
	}
	return v.([]any)
}
func goName(s string) string {
	var result strings.Builder
	for _, part := range strings.FieldsFunc(s, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
		runes := []rune(part)
		start := 0
		for end := 1; end <= len(runes); end++ {
			if end < len(runes) {
				startsWord := unicode.IsUpper(runes[end]) &&
					(unicode.IsLower(runes[end-1]) || unicode.IsDigit(runes[end-1]) ||
						(end+1 < len(runes) && unicode.IsLower(runes[end+1])))
				if !startsWord {
					continue
				}
			}
			word := string(runes[start:end])
			switch upper := strings.ToUpper(word); upper {
			case "API", "CWD", "FS", "HTTP", "HTTPS", "ID", "JSON", "MCP", "MIME", "SSE", "URI", "URL":
				result.WriteString(upper)
			default:
				result.WriteRune(unicode.ToUpper(runes[start]))
				result.WriteString(string(runes[start+1 : end]))
			}
			start = end
		}
	}
	if result.Len() == 0 {
		return "Variant"
	}
	return result.String()
}
func refName(n node) string { s, _ := n["$ref"].(string); return strings.TrimPrefix(s, "#/$defs/") }
func goType(n node) string {
	if definition := refName(n); definition != "" {
		return goName(definition)
	}
	if a := asArray(n["allOf"]); len(a) == 1 {
		return goType(asObject(a[0]))
	}
	t := n["type"]
	if a, ok := t.([]any); ok {
		for _, v := range a {
			if v != "null" {
				t = v
				break
			}
		}
	}
	switch t {
	case "string":
		return "string"
	case "integer":
		if n["format"] == "uint64" {
			return "uint64"
		}
		if n["format"] == "uint16" {
			return "uint16"
		}
		return "int64"
	case "number":
		return "float64"
	case "boolean":
		return "bool"
	case "array":
		return "[]" + goType(asObject(n["items"]))
	}
	if a := asArray(n["anyOf"]); len(a) == 2 {
		for _, v := range a {
			q := asObject(v)
			if q["type"] != "null" {
				return goType(q)
			}
		}
	}
	return "json.RawMessage"
}
func flattenObject(n node) node {
	p := map[string]any{}
	r := []any{}
	for _, v := range asArray(n["allOf"]) {
		q := asObject(v)
		if x := refName(q); x != "" {
			q = defs[x]
		}
		q = flattenObject(q)
		for k, v := range asObject(q["properties"]) {
			p[k] = v
		}
		r = append(r, asArray(q["required"])...)
	}
	for k, v := range asObject(n["properties"]) {
		p[k] = v
	}
	r = append(r, asArray(n["required"])...)
	z := node{}
	for k, v := range n {
		z[k] = v
	}
	z["properties"] = p
	z["required"] = r
	return z
}
func emitStruct(typeName string, schema node, definition string) {
	schema = flattenObject(schema)
	fmt.Fprintf(&out, "type %s struct {\n", typeName)
	required := map[string]bool{}
	for _, property := range asArray(schema["required"]) {
		required[property.(string)] = true
	}
	properties := asObject(schema["properties"])
	for _, field := range slices.Sorted(maps.Keys(properties)) {
		fieldType := goType(asObject(properties[field]))
		if !required[field] {
			fieldType = "Optional[" + fieldType + "]"
		}
		fmt.Fprintf(&out, "%s %s `json:%q`\n", goName(field), fieldType, field)
	}
	fmt.Fprintln(&out, "Extra map[string]json.RawMessage `json:\"-\"`\n}")
	fmt.Fprintf(&out, "func (v %s) MarshalJSON() ([]byte,error) { return marshalObject(%q,v) }\nfunc (v *%s) UnmarshalJSON(b []byte) error { type plain %s; var p plain; if err:=unmarshalObject(%q,b,&p);err!=nil{return err};*v=%s(p);return nil }\n", typeName, definition, typeName, typeName, definition, typeName)
}
func unionBranches(n node) []any {
	if a := asArray(n["oneOf"]); len(a) > 0 {
		return a
	}
	return asArray(n["anyOf"])
}
func isStringUnion(n node) bool {
	a := unionBranches(n)
	if len(a) == 0 {
		return false
	}
	for _, v := range a {
		if asObject(v)["type"] != "string" {
			return false
		}
	}
	return true
}
func generate(schema []byte) ([]byte, error) {
	var doc node
	if err := json.Unmarshal(schema, &doc); err != nil {
		return nil, err
	}
	defs = map[string]node{}
	for k, v := range asObject(doc["$defs"]) {
		defs[k] = asObject(v)
	}
	seen := map[string]bool{}
	var walk func(any)
	walk = func(v any) {
		switch x := v.(type) {
		case map[string]any:
			if r := refName(node(x)); r != "" && !seen[r] {
				seen[r] = true
				walk(map[string]any(defs[r]))
			}
			for _, v := range x {
				walk(v)
			}
		case []any:
			for _, v := range x {
				walk(v)
			}
		}
	}
	for _, r := range roots {
		seen[r] = true
		walk(map[string]any(defs[r]))
	}
	out.Reset()
	fmt.Fprintf(&out, "// Code generated by acpgen; DO NOT EDIT.\n// Source: agentclientprotocol/agent-client-protocol@%s\n// schema/v1/schema.json blob %s\npackage protocol\n\nimport (\"encoding/json\"\n", sourcePin.Commit, sourcePin.SchemaBlob)
	if seen["SessionConfigSelectOptions"] {
		fmt.Fprintln(&out, "\"bytes\"")
	}
	fmt.Fprintln(&out, ")")
	fmt.Fprintf(&out, "const (\nWireVersion = %d\nSchemaVersion = %q\nSourceCommit = %q\nSchemaBlob = %q\n)\n", sourcePin.WireVersion, sourcePin.SchemaVersion, sourcePin.Commit, sourcePin.SchemaBlob)
	for _, definition := range slices.Sorted(maps.Keys(seen)) {
		typeName := goName(definition)
		schema := defs[definition]
		alternatives := unionBranches(schema)
		if len(alternatives) > 0 && !isStringUnion(schema) {
			fmt.Fprintf(&out, "// %s holds one of its concrete variant types in Value.\ntype %s struct { Value any }\n", typeName, typeName)
			variantNames := []string{}
			for index, alternative := range alternatives {
				branch := asObject(alternative)
				caseName, _ := branch["title"].(string)
				for _, property := range slices.Sorted(maps.Keys(asObject(branch["properties"]))) {
					if tag, ok := asObject(asObject(branch["properties"])[property])["const"].(string); ok {
						caseName = tag
					}
				}
				if caseName == "" {
					if composition := asArray(branch["allOf"]); len(composition) == 1 {
						caseName = refName(asObject(composition[0]))
					}
				}
				if caseName == "" {
					caseName = fmt.Sprintf("Variant%d", index+1)
				}
				variantName := typeName + goName(caseName) + "Variant"
				variantNames = append(variantNames, variantName)
				if branch["type"] == "array" {
					fmt.Fprintf(&out, "type %s %s\n", variantName, goType(branch))
					emitScalar(variantName, fmt.Sprintf("%s/%d", definition, index))
				} else {
					merged := flattenObject(branch)
					properties := asObject(merged["properties"])
					for field, property := range asObject(schema["properties"]) {
						properties[field] = property
					}
					merged["required"] = append(asArray(merged["required"]), asArray(schema["required"])...)
					emitStruct(variantName, merged, fmt.Sprintf("%s/%d", definition, index))
				}
			}
			accepted := make([]string, 0, 2*len(variantNames))
			for _, variantName := range variantNames {
				accepted = append(accepted, variantName, "*"+variantName)
			}
			fmt.Fprintf(&out, "func (v %s) MarshalJSON() ([]byte,error) { switch v.Value.(type) {case %s: return marshalValue(%q,v.Value);default:return nil,errVariant} }\n", typeName, strings.Join(accepted, ","), definition)
			fmt.Fprintf(&out, "func (v *%s) UnmarshalJSON(b []byte) error {", typeName)
			if definition == "SessionConfigSelectOptions" {
				fmt.Fprintln(&out, "if bytes.Equal(bytes.TrimSpace(b),[]byte(\"[]\")) {var p SessionConfigSelectOptionsUngroupedVariant;if err:=json.Unmarshal(b,&p);err!=nil{return err};v.Value=p;return nil}")
			}
			fmt.Fprintf(&out, "i,err:=selectVariant(%q,b);if err!=nil{return err};switch i {", definition)
			for index, variantName := range variantNames {
				fmt.Fprintf(&out, "case %d: var p %s;if err:=json.Unmarshal(b,&p);err!=nil{return err};v.Value=p;", index, variantName)
			}
			fmt.Fprintln(&out, "};return nil }")
		} else if schema["type"] == "object" {
			emitStruct(typeName, schema, definition)
		} else {
			underlyingType := goType(schema)
			if isStringUnion(schema) || schema["enum"] != nil {
				underlyingType = "string"
			}
			fmt.Fprintf(&out, "type %s %s\n", typeName, underlyingType)
			emitScalar(typeName, definition)
			values := map[string]bool{}
			for _, value := range asArray(schema["enum"]) {
				if text, ok := value.(string); ok {
					values[text] = true
				}
			}
			if isStringUnion(schema) {
				for _, branch := range unionBranches(schema) {
					if text, ok := asObject(branch)["const"].(string); ok {
						values[text] = true
					}
				}
			}
			for _, value := range slices.Sorted(maps.Keys(values)) {
				fmt.Fprintf(&out, "const %s%s %s = %q\n", typeName, goName(value), typeName, value)
			}
		}
	}
	fmt.Fprintln(&out, "var supportedDefinitions = definitionSet([]string{")
	for _, nm := range slices.Sorted(maps.Keys(seen)) {
		fmt.Fprintf(&out, "%q,\n", nm)
	}
	fmt.Fprintln(&out, "})")
	return format.Source(out.Bytes())
}
func emitScalar(typeName, definition string) {
	fmt.Fprintf(&out, "func (v %s) MarshalJSON() ([]byte,error) { type plain %s; return marshalValue(%q,plain(v)) }\n", typeName, typeName, definition)
	fmt.Fprintf(&out, "func (v *%s) UnmarshalJSON(b []byte) error {if err:=ValidateJSON(%q,b);err!=nil{return err};type plain %s; var p plain;if err:=unmarshalExact(b,&p);err!=nil{return err};*v=%s(p);return nil }\n", typeName, definition, typeName, typeName)
}
func checkBlob(b []byte, want string) error {
	h := sha1.New()
	_, _ = fmt.Fprintf(h, "blob %d\x00", len(b))
	_, _ = h.Write(b)
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("source pin mismatch: got %s want %s", got, want)
	}
	return nil
}
func main() {
	meta, err := os.ReadFile("meta.json")
	if err != nil {
		panic(err)
	}
	if err = checkBlob(meta, sourcePin.MetadataBlob); err != nil {
		panic(err)
	}
	b, err := os.ReadFile("schema.json")
	if err != nil {
		panic(err)
	}
	if err = checkBlob(b, sourcePin.SchemaBlob); err != nil {
		panic(err)
	}
	b, err = generate(b)
	if err != nil {
		panic(err)
	}
	if err = os.WriteFile("types_gen.go", b, 0644); err != nil {
		panic(err)
	}
}
