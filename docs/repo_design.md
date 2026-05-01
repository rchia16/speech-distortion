# Repo Design

## Principles

- keep every layer replaceable
- prefer typed contracts over implicit dictionaries
- preserve speaker identity by default
- treat timing control as a first-class module
- separate planning from rendering

## Module responsibilities

### `alignment`
Consumes waveform and transcript and returns word- and phone-level alignment with confidence metadata.

### `phonology`
Builds a phone graph and attaches phonetic features, syllable structure, stress, and articulatory complexity.

### `planning`
Transforms phone-level representations into an edit plan that specifies what to change and where.

### `editing`
Applies source-domain local transformations for distortions and local duration edits.

### `resynthesis`
Generates replacement fragments for additions and substitutions that cannot be produced from source editing alone.

### `timbre`
Projects synthetic fragments back toward the source speaker.

### `timing`
Allocates local duration increases and compensatory shortening while preserving utterance length.

### `stitching`
Assembles original and edited fragments into the final waveform.

### `orchestration`
Coordinates the full pass and owns the high-level execution order.

## Suggested implementation order

1. models
2. config loading
3. alignment
4. phonology
5. planning
6. timing
7. editing
8. stitching
9. resynthesis
10. timbre
11. full orchestration
