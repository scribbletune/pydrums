<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# PyDrums Project Instructions

This is a music AI project for drum pattern generation and MIDI conversion. Please follow these guidelines when working with this codebase:

## Project Overview

- **Purpose**: AI-powered drum pattern generation using Ollama and few-shot learning
- **Architecture**: Python package with modular components for data loading, pattern generation, and MIDI conversion
- **Target Users**: Musicians, producers, and developers working with drum patterns and MIDI

## Code Style Guidelines

- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Include comprehensive docstrings for all classes and functions
- Use descriptive variable names that reflect the music domain (e.g., `tempo_bpm`, `drum_patterns`, `midi_note`)

## Music Domain Knowledge

- **Drum Notation**: Use the 6-token system (x, -, R, \_, [, ]) for pattern representation
  - `x` = hit/strike
  - `-` = rest/silence
  - `R` = roll
  - `_` = ghost note
  - `[` = flam start
  - `]` = flam end
- **Drum Abbreviations**: ch (closed hi-hat), sd (snare), bd (bass drum), oh (open hi-hat), etc.
- **MIDI Standards**: Use General MIDI drum mapping (channel 9, note 36=kick, 38=snare, etc.)

## AI/ML Considerations

- This project uses **few-shot learning** with Ollama, not traditional model training
- Training data consists of professional drum patterns from GitHub repositories
- Pattern generation should validate output format and provide fallback mechanisms
- Temperature and sampling parameters should be tunable for creative control

## Error Handling

- Always provide fallback patterns when AI generation fails
- Validate pattern format before MIDI conversion
- Include helpful error messages for musicians who may not be technical

## Testing Focus Areas

- Pattern format validation
- MIDI file generation accuracy
- Training data processing integrity
- Cross-platform compatibility (especially for MIDI playback)

## Dependencies

- Core: `ollama`, `mido`, `requests`, `pandas`
- Optional: `torch`, `transformers` (for future model training)
- Audio: `librosa`, `music21` (for advanced audio processing)

## File Organization

- `src/pydrums/`: Main package code
- `data/`: Training data and pattern collections
- `midi_output/`: Generated MIDI files
- `examples/`: Usage examples and demos
- `tests/`: Unit tests and integration tests

When suggesting code improvements or new features, consider the musical context and ensure compatibility with standard music production workflows.
