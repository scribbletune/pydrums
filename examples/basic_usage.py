#!/usr/bin/env python3
"""
Example: Basic pattern generation and MIDI conversion
"""

from pydrums import PatternGenerator, MidiConverter, DataLoader

def main():
    print("🎵 PyDrums Example: Basic Usage")
    print("=" * 50)
    
    # Setup data if needed
    print("1. Setting up data...")
    loader = DataLoader()
    
    # Use the DrumMachinePatterns260 source (recommended)
    patterns = loader.load_drum_machine_patterns_260()
    
    if not patterns:
        print("   Downloading training data...")
        patterns = loader.load_drum_machine_patterns_260(force_reload=True)
    
    training_data = loader.convert_patterns_to_training_data(patterns)
    loader.save_training_data(training_data)
    
    print(f"   ✅ Ready with {len(training_data)} training examples")
    print(f"   📊 Dataset: {len(patterns)} patterns across 17 musical styles")
    
    # Generate patterns from different styles
    print("\\n2. Generating patterns from various styles...")
    generator = PatternGenerator()
    
    test_descriptions = [
        "Create a funk groove with ghost notes",
        "Generate an afro-cuban rhythm",
        "Make a reggae one drop pattern", 
        "Create a jazz shuffle beat",
        "Generate a disco four-on-the-floor pattern",
        "Make a rock ballad pattern"
    ]
    
    generated_patterns = []
    
    for desc in test_descriptions:
        print(f"   🎵 {desc}")
        pattern = generator.generate_pattern(desc)
        generated_patterns.append(pattern)
        print(f"      Pattern: {pattern['pattern_line'][:50]}...")
        print(f"      Valid: {pattern['is_valid']}")
        if pattern.get('validation_notes'):
            print(f"      Notes: {', '.join(pattern['validation_notes'])}")
    
    # Convert to MIDI
    print("\\n3. Converting to MIDI...")
    converter = MidiConverter()
    
    midi_files = converter.batch_convert(generated_patterns, tempo_bpm=120, loop_count=4)
    
    print(f"   ✅ Created {len(midi_files)} MIDI files:")
    for midi_file in midi_files:
        print(f"      📄 {midi_file.name}")
    
    print("\\n🎉 Example complete! Check the 'midi_output' folder for your MIDI files.")
    print("\\n📚 Your dataset includes 17 musical styles:")
    print("   Funk, Rock, Disco, Reggae, Jazz, Ballad, Pop, R&B, Latin,")
    print("   Afro-Cuban, Shuffle, Bossa Nova, Blues, Waltz, March, Tango, and General")

if __name__ == "__main__":
    main()
