# PyDrums Dataset Documentation

## Overview

PyDrums uses a comprehensive dataset of professional drum patterns to train AI models for pattern generation. This document provides detailed information about the current dataset, sources, and capabilities.

## Current Dataset Statistics

- **Total Patterns**: 268 professional drum patterns
- **Training Examples**: 1,331 AI training pairs
- **Musical Styles**: 17 different genres
- **Time Signatures**: Multiple (4/4, 12/8, 3/4, etc.)
- **Pattern Format**: 6-token notation system (x, -, R, \_, [, ])

## Detailed Style Breakdown

### 1. Funk (150 examples)

- **Characteristics**: Syncopated rhythms, heavy use of ghost notes, emphasis on groove
- **Common Patterns**: Off-beat hi-hats, snare on beats 2 and 4, syncopated kick patterns
- **Example Patterns**: "AfroCub1", "Funk1", "Funk2"

### 2. Rock (135 examples)

- **Characteristics**: Steady backbeat, driving rhythm, straightforward patterns
- **Common Patterns**: Hi-hat on eighth notes, snare on 2 and 4, kick on 1 and 3
- **Example Patterns**: "Rock1", "Rock2", various rock subdivisions

### 3. General (122 examples)

- **Characteristics**: Versatile patterns suitable for multiple genres
- **Common Patterns**: Basic drum set combinations, foundational rhythms
- **Usage**: Fallback patterns, learning examples, versatile grooves

### 4. Disco (110 examples)

- **Characteristics**: Four-on-the-floor kick pattern, steady hi-hat, danceable groove
- **Common Patterns**: Kick on every quarter note, snare on 2 and 4, open hi-hat accents
- **Era**: 1970s dance music, modern house influences

### 5. Reggae (105 examples)

- **Characteristics**: One drop rhythm, emphasis on beat 3, laid-back feel
- **Common Patterns**: Kick and snare on beat 3, rim shots, syncopated hi-hat
- **Variations**: One drop, rockers, steppers

### 6. Jazz (90 examples)

- **Characteristics**: Swing feel, brush techniques, subtle ghost notes
- **Common Patterns**: Swung eighth notes, ride cymbal patterns, brush rolls
- **Sub-genres**: Swing, bebop, Latin jazz influences

### 7. Ballad (90 examples)

- **Characteristics**: Slow tempo, emotional expression, space between notes
- **Common Patterns**: Simple kick and snare, emphasis on dynamics, minimal hi-hat
- **Usage**: Slow songs, emotional moments, sparse arrangements

### 8. Pop (90 examples)

- **Characteristics**: Radio-friendly, accessible rhythms, clear structure
- **Common Patterns**: Straightforward backbeat, consistent hi-hat, commercial appeal
- **Era**: Modern pop music, chart-friendly patterns

### 9. R&B (90 examples)

- **Characteristics**: Groove-oriented, soul influences, sophisticated rhythms
- **Common Patterns**: Syncopated patterns, ghost notes, pocket playing
- **Sub-genres**: Classic soul, modern R&B, neo-soul

### 10. Latin (85 examples)

- **Characteristics**: Complex polyrhythms, percussion influences, cultural authenticity
- **Common Patterns**: Clave patterns, montuno rhythms, cross-rhythms
- **Sub-genres**: Salsa, mambo, cha-cha, general Latin

### 11. Afro-Cuban (75 examples)

- **Characteristics**: Traditional African-Cuban rhythms, complex polyrhythms
- **Common Patterns**: Son clave, rumba patterns, traditional percussion translations
- **Cultural Context**: Authentic Cuban musical traditions

### 12. Shuffle (45 examples)

- **Characteristics**: Swung eighth note feel, blues influences
- **Common Patterns**: Triplet-based subdivisions, blues shuffle, Texas shuffle
- **Usage**: Blues, country, rock variations

### 13. Bossa Nova (45 examples)

- **Characteristics**: Brazilian jazz influences, subtle Latin groove
- **Common Patterns**: Gentle Latin feel, jazz harmony support, sophisticated rhythm
- **Origin**: Brazilian music, João Gilberto influence

### 14. Blues (45 examples)

- **Characteristics**: Traditional blues rhythms, 12-bar support
- **Common Patterns**: Shuffle feel, straight eighth notes, call and response
- **Sub-genres**: Delta blues, Chicago blues, modern blues

### 15. Waltz (24 examples)

- **Characteristics**: 3/4 time signature, classical influences
- **Common Patterns**: Strong beat 1, lighter beats 2 and 3, flowing feel
- **Usage**: Classical music, folk dances, ballroom dancing

### 16. March (20 examples)

- **Characteristics**: Military precision, steady tempo, ceremonial feel
- **Common Patterns**: Strong downbeats, snare drum rolls, parade rhythm
- **Usage**: Military ceremonies, parades, patriotic music

### 17. Tango (10 examples)

- **Characteristics**: Argentinian tango rhythm, dramatic pauses
- **Common Patterns**: Distinctive tango clave, staccato accents, passionate expression
- **Origin**: Argentinian dance music

## Data Sources

### Primary Source: DrumMachinePatterns260

- **URL**: `https://github.com/stephenhandley/DrumMachinePatterns/blob/master/Sources/DrumMachinePatterns260/Patterns.json`
- **Format**: JSON with structured pattern data
- **Content**: 268 professional drum machine patterns
- **Structure**: Each pattern includes title, signature, length, and track data
- **Quality**: High-quality, professionally curated patterns

### Data Structure Example

```json
{
  "title": "AfroCub1",
  "signature": "4/4",
  "length": 16,
  "tracks": {
    "ClosedHiHat": ["Note", "Rest", "Note", "Note", ...],
    "SnareDrum": ["Rest", "Rest", "Rest", "Rest", ...],
    "BassDrum": ["Note", "Rest", "Rest", "Rest", ...]
  }
}
```

### Legacy Sources

- **stephenhandley/DrumMachinePatterns**: Individual JSON files (API access issues)
- **montoyamoraga/drum-machine-patterns**: Markdown format patterns (reference)

## Pattern Notation System

PyDrums uses a 6-token notation system for representing drum patterns:

- `x` = Hit/strike the drum (main accent)
- `-` = Rest/silence (no sound)
- `R` = Roll (extended sound, multiple rapid hits)
- `_` = Ghost note (quiet hit, subtle accent)
- `[` = Flam start (grace note before main hit)
- `]` = Flam end (completion of flam)

### Drum Abbreviations

- `ch` = Closed Hi-Hat
- `oh` = Open Hi-Hat
- `sd` = Snare Drum
- `bd` = Bass Drum (Kick)
- `hh` = Hi-Hat Pedal
- `cc` = Crash Cymbal
- `rc` = Ride Cymbal
- `ht` = High Tom
- `mt` = Mid Tom
- `lt` = Low Tom

### Pattern Format

```
ch: x-x-x-x-x-x-x-x-; sd: ----x-------x---; bd: x-----x-x-------
```

## Training Data Generation

Each source pattern generates multiple training examples:

1. **Style-based prompts**: "Create a [style] drum pattern"
2. **Action-based prompts**: "Generate a [style] beat"
3. **Format-based prompts**: "Make a [style] drum loop"
4. **Time signature prompts**: "Create a [time_sig] [style] pattern"

Example training pair:

```json
{
  "input": "Create a funk drum pattern",
  "output": "ch: x-x-x-x-x-x-x-x-; sd: ----x--x----x---; bd: x-----x---x-----",
  "style": "funk",
  "time_signature": "4/4",
  "source_pattern": "Funk1"
}
```

## Adding New Data Sources

PyDrums supports multiple data sources. To add a new source:

### 1. JSON Format Requirements

```json
{
  "patterns": [
    {
      "name": "PatternName",
      "style": "funk",
      "timeSignature": "4/4",
      "drums": {
        "BassDrum": ["Note", "Rest", "Note", "Rest"],
        "SnareDrum": ["Rest", "Note", "Rest", "Note"]
      }
    }
  ]
}
```

### 2. CLI Command

```bash
pydrums setup-data --additional-url "https://example.com/patterns.json" --additional-name "my_source"
```

### 3. Supported Formats

- **Format A**: `drums` key with drum names
- **Format B**: `tracks` key with drum names (DrumMachinePatterns260)
- **Custom**: Extend `DataLoader` class for other formats

## Performance Metrics

### Generation Quality

- **Valid Pattern Rate**: ~85% of generated patterns follow correct format
- **Style Accuracy**: High correlation between requested style and output
- **Musical Quality**: Based on professional drum machine patterns

### Dataset Coverage

- **Time Signatures**: 4/4 (majority), 12/8, 3/4, 6/8
- **Tempo Range**: Patterns suitable for 60-180 BPM
- **Complexity**: From simple 4-beat patterns to complex 16-beat patterns

## Technical Implementation

### Few-Shot Learning

- Uses 3-5 relevant examples for each generation
- Selects examples based on keyword matching and style hints
- Fallback to random examples if no matches found

### Pattern Validation

- Checks for proper drum notation format
- Validates drum abbreviations against known mapping
- Ensures pattern contains both hits and rests

### MIDI Conversion

- Uses General MIDI drum mapping (channel 9)
- Supports all 6 notation tokens
- Configurable tempo, loop count, and resolution

## Future Enhancements

### Planned Features

1. **Real-time training**: Add patterns from user feedback
2. **Style transfer**: Convert patterns between styles
3. **Complexity control**: Generate simple or complex variations
4. **Humanization**: Add timing and velocity variations

### Additional Data Sources

1. **Live recording data**: Real drummer performances
2. **Genre-specific collections**: Specialized pattern libraries
3. **User contributions**: Community-driven pattern sharing
4. **Historical patterns**: Classic drum machine patterns from vintage hardware

---

_Last updated: July 13, 2025_
_Dataset version: v1.0 with DrumMachinePatterns260 integration_
