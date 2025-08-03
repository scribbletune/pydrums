# PyDrums 🥁

**AI-powered drum pattern generation and MIDI conversion**

PyDrums is a comprehensive Python toolkit that uses AI (via Ollama) to generate professional drum patterns from text descriptions and convert them to MIDI files. Perfect for musicians, producers, and developers working with rhythm and percussion.

## ✨ Features

- 🤖 **AI Pattern Generation**: Uses few-shot learning with Ollama to generate drum patterns from natural language
- 🎼 **MIDI Conversion**: Convert patterns to standard MIDI files compatible with any DAW
- 📚 **Professional Training Data**: Built on 200+ professional drum patterns across 15+ musical styles
- 🔄 **Multiple Data Sources**: Load patterns from GitHub repositories and custom JSON sources
- 🎮 **Interactive Mode**: Real-time pattern generation with immediate feedback
- 🛠️ **CLI Tool**: Command-line interface for batch processing and automation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pydrums.git
cd pydrums

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -e .
```

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull a model**: `ollama pull llama3.1:latest`

### Basic Usage

```bash
# 1. Setup training data (downloads 268 professional patterns with speed variations)
pydrums setup-data --skip-primary

# 2. Generate a pattern from 17 available styles with speed control (automatically creates MIDI)
pydrums generate -d "Create a funky afro-cuban beat"

# 3. Interactive mode - try any of the 17 styles with speed variations!
pydrums generate --interactive

# 4. Convert pattern string to MIDI
pydrums convert -p "ch: x-x-x-x-; sd: ----x---; bd: x-----x-"

# 5. Regenerate training data with enhanced speed variations
pydrums regenerate-training

# 6. Check your expanded dataset
pydrums info
```

## 📖 Documentation

### Pattern Notation

PyDrums uses a 6-token notation system:

- `x` = Hit/strike the drum
- `-` = Rest/silence
- `R` = Roll (extended sound)
- `_` = Ghost note (quiet hit)
- `[` = Flam start (grace note)
- `]` = Flam end

### Drum Abbreviations

- `ch` = Closed Hi-Hat
- `oh` = Open Hi-Hat
- `sd` = Snare Drum
- `bd` = Bass Drum
- `hh` = Hi-Hat Pedal
- `cc` = Crash Cymbal
- `rc` = Ride Cymbal
- `ht` = High Tom
- `mt` = Mid Tom
- `lt` = Low Tom

### Additional Documentation

- 📊 **[DATASET.md](DATASET.md)**: Comprehensive dataset documentation with detailed style breakdowns and statistics
- 🎵 **[examples/](examples/)**: Usage examples and tutorials
- 🛠️ **[.github/copilot-instructions.md](.github/copilot-instructions.md)**: Development guidelines

### Python API

```python
from pydrums import PatternGenerator, MidiConverter, DataLoader

# Generate patterns
generator = PatternGenerator()
pattern = generator.generate_pattern("Create a jazz shuffle")

# Convert to MIDI
converter = MidiConverter()
midi_file = converter.pattern_to_midi(pattern, tempo_bpm=140)

# Load additional data
loader = DataLoader()
patterns = loader.load_github_json_patterns()
```

## 🎯 Use Cases

- **Music Production**: Generate drum patterns for songs and beats
- **Practice**: Create backing tracks for musicians
- **Game Development**: Generate dynamic music for games
- **Music Education**: Learn about rhythm and pattern construction
- **AI Research**: Experiment with music generation and few-shot learning

## 🔧 Configuration

### Data Source Management

```bash
# Setup with all available sources
pydrums setup-data

# Use only DrumMachinePatterns260 (recommended)
pydrums setup-data --skip-primary

# Force re-download of all data
pydrums setup-data --force-reload

# Add your own JSON pattern source
pydrums setup-data --additional-url "https://raw.githubusercontent.com/user/repo/patterns.json" --additional-name "my_patterns"
```

### Available Style Examples

```bash
# Generate patterns from different styles in your dataset
pydrums generate -d "Create a funk groove with ghost notes"
pydrums generate -d "Make an afro-cuban pattern"
pydrums generate -d "Generate a reggae one drop beat"
pydrums generate -d "Create a jazz shuffle pattern"
pydrums generate -d "Make a disco four-on-the-floor beat"
pydrums generate -d "Generate a bossa nova rhythm"
pydrums generate -d "Create a rock ballad pattern"

# NEW: Speed variation examples
pydrums generate -d "Create a half-time funk groove"
pydrums generate -d "Generate a double-time rock beat"
pydrums generate -d "Make a simple quarter note disco pattern"
pydrums generate -d "Create a laid-back jazz groove"
```

### Model Selection

```bash
# Use different Ollama model
pydrums generate -d "rock beat" -m "mistral:latest"
```

### Advanced Options

```python
# Custom generation parameters
pattern = generator.generate_pattern(
    "Create a complex progressive pattern",
    num_examples=5,        # More examples for better context
    temperature=0.8,       # Higher creativity
    style_hint="funk"      # Guide style selection from 17 available styles
)

# MIDI with custom settings
midi_file = converter.pattern_to_midi(
    pattern,
    tempo_bpm=140,
    loop_count=8,          # 8 repetitions
    ticks_per_beat=960     # Higher resolution
)
```

## 📊 Training Data

PyDrums includes professionally curated drum patterns from multiple sources:

### Current Dataset Statistics

- **268 Professional Patterns**: High-quality drum machine patterns
- **1,331+ AI Training Examples**: Generated from professional patterns with speed variations
- **17 Musical Styles**: Comprehensive coverage of musical genres
- **4 Speed Variations**: Normal, half-time, double-time, and quarter-note patterns
- **Multiple Time Signatures**: 4/4, 12/8, 3/4, and more
- **JSON Format**: Structured data for reliable AI training

### Speed Variations Available

- **Normal Speed** (16th notes): Standard drum patterns with 16-character resolution
- **Half-Time** (32nd notes): Slower, laid-back grooves with extended spacing
- **Double-Time** (8th notes): Fast, energetic patterns with rapid hits
- **Quarter Notes**: Simple, minimal patterns emphasizing strong beats

### Pattern Length Examples

```
Normal (16 chars):    ch: x-x-x-x-x-x-x-x-; bd: x---x---x---x---
Half-time (32 chars): ch: x---x---x---x---x---x---x---x---; bd: x-------x-------x-------x-------
Double-time (8 chars): ch: xxxxxxxx; bd: x-x-x-x-
Quarter notes (4 chars): ch: xxxx; bd: xxxx
```

### Available Musical Styles

1. **Funk** (150 examples) - Syncopated grooves and ghost notes
2. **Rock** (135 examples) - Classic and modern rock beats
3. **General** (122 examples) - Versatile patterns for any genre
4. **Disco** (110 examples) - Four-on-the-floor dance patterns
5. **Reggae** (105 examples) - One drop and rockers patterns
6. **Jazz** (90 examples) - Swing and shuffle patterns
7. **Ballad** (90 examples) - Slow, emotional patterns
8. **Pop** (90 examples) - Commercial and radio-friendly beats
9. **R&B** (90 examples) - Soul and rhythm & blues grooves
10. **Latin** (85 examples) - Salsa, mambo, and cha-cha patterns
11. **Afro-Cuban** (75 examples) - Traditional African-Cuban rhythms
12. **Shuffle** (45 examples) - Swung eighth note patterns
13. **Bossa Nova** (45 examples) - Brazilian jazz patterns
14. **Blues** (45 examples) - Traditional blues rhythms
15. **Waltz** (24 examples) - 3/4 time signature patterns
16. **March** (20 examples) - Military and ceremonial beats
17. **Tango** (10 examples) - Argentinian tango rhythms

### Data Sources

1. **Primary**: [DrumMachinePatterns260](https://github.com/stephenhandley/DrumMachinePatterns/blob/master/Sources/DrumMachinePatterns260/Patterns.json) - 268 patterns in perfect JSON format
2. **Legacy**: [stephenhandley/DrumMachinePatterns](https://github.com/stephenhandley/DrumMachinePatterns) - Individual pattern files
3. **Reference**: [montoyamoraga/drum-machine-patterns](https://github.com/montoyamoraga/drum-machine-patterns) - Additional patterns in markdown format

## 🧠 How It Works

PyDrums uses **few-shot learning** rather than traditional model training:

1. **Context Loading**: Selects relevant example patterns based on your description
2. **Prompt Engineering**: Creates structured prompts with notation guides and examples
3. **AI Generation**: Uses Ollama to generate new patterns following the established format
4. **Validation**: Checks output format and provides fallbacks if needed
5. **MIDI Conversion**: Converts text patterns to standard MIDI using General MIDI mapping

This approach is:

- ⚡ **Fast**: No training time required
- 🎯 **Accurate**: Uses proven professional patterns as examples
- 🔄 **Adaptable**: Easy to add new styles and patterns
- 💻 **Local**: Runs entirely on your machine with Ollama

## 🛠️ Development

### Project Structure

```
pydrums/
├── src/pydrums/           # Main package
│   ├── __init__.py
│   ├── pattern_generator.py    # AI pattern generation
│   ├── midi_converter.py       # MIDI file creation
│   ├── data_loader.py          # Data loading and processing
│   └── cli.py                  # Command-line interface
├── data/                       # Training data
├── midi_output/               # Generated MIDI files
├── examples/                  # Usage examples
├── tests/                     # Unit tests
└── requirements.txt           # Dependencies
```

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Adding New Pattern Sources

To add a new pattern data source:

1. Implement loading logic in `DataLoader.load_additional_json_source()`
2. Add conversion logic for the specific format
3. Update the CLI to accept the new source
4. Add tests for the new functionality

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local AI capabilities
- [stephenhandley/DrumMachinePatterns](https://github.com/stephenhandley/DrumMachinePatterns) for excellent training data
- [montoyamoraga/drum-machine-patterns](https://github.com/montoyamoraga/drum-machine-patterns) for additional pattern resources
- The music production community for feedback and inspiration

## 📞 Support

- 🐛 [Report bugs](https://github.com/yourusername/pydrums/issues)
- 💡 [Request features](https://github.com/yourusername/pydrums/issues)
- 📧 [Email support](mailto:your.email@example.com)
- 💬 [Community discussions](https://github.com/yourusername/pydrums/discussions)

---

**Happy drumming! 🥁🎵**
