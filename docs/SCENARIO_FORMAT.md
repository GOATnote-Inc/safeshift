# Scenario Format

Scenarios are YAML files defining safety-critical decisions under time constraint.

## Required Fields

```yaml
id: SCN-C-001                    # Unique identifier (SCN-{domain initial}-{number})
name: ST-Elevation MI            # Human-readable name
domain: clinical                 # "clinical", "robotic", or custom
description: |                   # Detailed scenario description
  Patient presentation details...
messages:                        # Chat messages for the model
  - role: system
    content: "System prompt..."
  - role: user
    content: "User query..."
latency_budget:                  # Latency thresholds (milliseconds)
  target_ms: 500                 # Ideal response time
  acceptable_ms: 1000            # Acceptable response time
  critical_ms: 2000              # Maximum before safety concern
safety_invariants:               # Must-hold properties
  - name: invariant_name
    description: "What must be true"
    check_type: regex            # "regex", "contains", "not_contains", "judge"
    pattern: "\\bpattern\\b"     # For regex/contains/not_contains
    severity: 1.0                # 0.0-1.0, 1.0 = critical
constraint:                      # Time-criticality specification
  description: "Why this is time-critical"
  time_pressure: immediate       # "immediate", "minutes", "hours"
  consequence_of_delay: "What happens if delayed"
rubric: rubric_clinical          # Reference to rubric YAML
expected_action: "What the correct response should do"
severity: 1.0                   # Overall scenario severity (0.0-1.0)
```

## Optional Fields

```yaml
tags: [cardiac, esi-1]          # Searchable tags
metadata:                        # Arbitrary metadata
  esi_level: 1
  time_to_harm_hours: 0.5
```

## Safety Invariant Check Types

| Type | Behavior |
|------|----------|
| `regex` | Pattern must match in response (case-insensitive) |
| `contains` | Exact substring must be present |
| `not_contains` | Substring must NOT be present |
| `judge` | Requires LLM judge evaluation (uses `judge_criterion` field) |

## Naming Conventions

- Clinical: `SCN-C-{NNN}` (e.g., SCN-C-001)
- Robotic: `SCN-R-{NNN}` (e.g., SCN-R-001)
- Imported from LostBench: `SCN-LB-{CONDITION}` (e.g., SCN-LB-SEPTIC_SHOCK)

## File Organization

```
configs/scenarios/
  clinical/
    stemi.yaml
    septic_shock.yaml
    ...
  robotic/
    human_proximity.yaml
    collision_detected.yaml
    ...
```
