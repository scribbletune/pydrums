"""
Command-line interface for PyDrums
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .data_loader import DataLoader
from .pattern_generator import PatternGenerator  
from .midi_converter import MidiConverter


def setup_data_command(args):
    """Set up training data from GitHub sources"""
    print("🔄 Setting up training data...")
    
    loader = DataLoader(args.data_dir)
    all_patterns = []
    
    # Load primary GitHub source (stephenhandley patterns)
    if not args.skip_primary:
        print("📥 Loading primary source (stephenhandley/DrumMachinePatterns)...")
        patterns = loader.load_github_json_patterns(force_reload=args.force_reload)
        all_patterns.extend(patterns)
        print(f"   ✅ Loaded {len(patterns)} patterns from primary source")
    
    # Load DrumMachinePatterns260 source
    if not args.skip_260:
        print("📥 Loading DrumMachinePatterns260 source...")
        patterns_260 = loader.load_drum_machine_patterns_260(force_reload=args.force_reload)
        all_patterns.extend(patterns_260)
        print(f"   ✅ Loaded {len(patterns_260)} patterns from DrumMachinePatterns260")
    
    # Load additional source if provided
    if args.additional_url:
        print(f"📥 Loading additional source: {args.additional_url}")
        additional_data = loader.load_additional_json_source(
            args.additional_url, 
            args.additional_name or "additional_source"
        )
        all_patterns.extend(additional_data)
        print(f"   ✅ Loaded {len(additional_data)} patterns from additional source")
    
    if not all_patterns:
        print("❌ No patterns loaded. Please check your sources.")
        return
    
    print(f"\n📊 Total patterns collected: {len(all_patterns)}")
    
    # Convert to training data
    print("🔄 Converting patterns to training data with speed variations...")
    training_data = loader.convert_patterns_to_training_data(all_patterns)
    
    # Save training data
    loader.save_training_data(training_data, "training_data.json")
    
    # Show statistics
    stats = loader.get_data_statistics(training_data)
    print("\n📊 TRAINING DATA STATISTICS:")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Unique styles: {stats['unique_styles']}")
    print(f"Style distribution: {stats['style_distribution']}")
    
    # Show speed distribution
    speed_distribution = {}
    for example in training_data:
        speed = example.get('speed', 'normal')
        speed_distribution[speed] = speed_distribution.get(speed, 0) + 1
    
    print(f"Speed distribution: {speed_distribution}")
    print("\n🎯 Speed variations now available:")
    print("  - Normal speed (16th notes)")
    print("  - Half-time (32nd notes, slower feel)")  
    print("  - Double-time (8th notes, faster feel)")
    print("  - Quarter notes (simple, strong beats)")


def regenerate_training_data_command(args):
    """Regenerate training data with enhanced speed variations"""
    print("🔄 Regenerating training data with speed variations...")
    
    loader = DataLoader(args.data_dir)
    
    # Load existing patterns  
    patterns_260 = loader.load_drum_machine_patterns_260()
    print(f"📊 Loaded {len(patterns_260)} base patterns")
    
    # Convert to enhanced training data
    training_data = loader.convert_patterns_to_training_data(patterns_260)
    
    # Save enhanced training data
    loader.save_training_data(training_data, "training_data_enhanced.json")
    
    # Show enhanced statistics
    stats = loader.get_data_statistics(training_data)
    speed_distribution = {}
    pattern_length_distribution = {}
    
    for example in training_data:
        speed = example.get('speed', 'normal')
        speed_distribution[speed] = speed_distribution.get(speed, 0) + 1
        
        length = example.get('pattern_length', 16)
        pattern_length_distribution[length] = pattern_length_distribution.get(length, 0) + 1
    
    print("\n🎉 ENHANCED TRAINING DATA GENERATED:")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Speed distribution: {speed_distribution}")
    print(f"Pattern lengths: {sorted(pattern_length_distribution.keys())}")
    print("\n✅ You can now generate patterns with speed variations!")
    print("Examples:")
    print('  pydrums generate -d "Create a half-time funk groove" --to-midi')
    print('  pydrums generate -d "Generate a double-time rock beat" --to-midi')
    print('  pydrums generate -d "Make a simple quarter note disco pattern" --to-midi')


def generate_command(args):
    """Generate drum patterns from text descriptions"""
    print("🎵 Generating drum patterns...")
    
    generator = PatternGenerator(args.model, args.data_dir)
    
    if args.interactive:
        generator.interactive_mode()
        return
    
    if args.description:
        descriptions = [args.description]
    elif args.batch_file:
        # Read descriptions from file
        batch_path = Path(args.batch_file)
        if not batch_path.exists():
            print(f"❌ Batch file not found: {batch_path}")
            return
        
        with open(batch_path, 'r') as f:
            descriptions = [line.strip() for line in f if line.strip()]
    else:
        print("❌ Please provide either --description or --batch-file")
        return
    
    # Generate patterns
    results = generator.batch_generate(
        descriptions,
        num_examples=args.examples,
        temperature=args.temperature
    )
    
    # Display results with speed information
    for i, result in enumerate(results, 1):
        print(f"\n🎵 Pattern {i}: {result['description']}")
        print(f"   Pattern: {result['pattern_line']}")
        print(f"   Valid: {result['is_valid']}")
        detected_speed = result.get('detected_speed')
        if detected_speed and detected_speed != 'normal':
            print(f"   🎯 Detected Speed: {detected_speed}")
        else:
            print(f"   Speed: normal (16th notes)")
        
        # Show if random examples were used
        if result.get('used_random_fallback', False):
            print(f"   ⚠️  Used random examples (no keyword matches found)")
    
    # Convert to MIDI if requested
    if args.to_midi:
        converter = MidiConverter(args.output_dir)
        midi_files = converter.batch_convert(
            results,
            tempo_bpm=args.tempo,
            loop_count=args.loops,
            include_tempo=args.include_tempo
        )
        print(f"\n🎼 Created {len(midi_files)} MIDI files with speed-appropriate timing")
        if not args.include_tempo:
            print("   ℹ️  MIDI files are tempo-neutral (no tempo metadata)")
    
    # Save results
    if args.save_patterns:
        output_path = Path(args.output_dir) / "generated_patterns.json"
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved patterns: {output_path}")


def convert_command(args):
    """Convert patterns to MIDI"""
    print("🎼 Converting patterns to MIDI...")
    
    converter = MidiConverter(args.output_dir)
    
    if args.pattern:
        # Convert single pattern string
        output_path = converter.pattern_string_to_midi(
            args.pattern,
            args.description or "custom_pattern",
            tempo_bpm=args.tempo,
            loop_count=args.loops,
            include_tempo=args.include_tempo
        )
        print(f"✅ Created: {output_path}")
        if not args.include_tempo:
            print("   ℹ️  MIDI file is tempo-neutral (no tempo metadata)")
    
    elif args.patterns_file:
        # Convert patterns from file
        import json
        patterns_path = Path(args.patterns_file)
        
        if not patterns_path.exists():
            print(f"❌ Patterns file not found: {patterns_path}")
            return
        
        with open(patterns_path, 'r') as f:
            patterns = json.load(f)
        
        midi_files = converter.batch_convert(
            patterns,
            tempo_bpm=args.tempo,
            loop_count=args.loops,
            include_tempo=args.include_tempo
        )
        print(f"✅ Created {len(midi_files)} MIDI files")
        if not args.include_tempo:
            print("   ℹ️  MIDI files are tempo-neutral (no tempo metadata)")


def info_command(args):
    """Show information about data and models"""
    print("ℹ️  PyDrums Information")
    print("=" * 50)
    
    # Check data
    loader = DataLoader(args.data_dir)
    training_data = loader.load_training_data()
    
    if training_data:
        stats = loader.get_data_statistics(training_data)
        print("📚 TRAINING DATA:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Styles: {list(stats['style_distribution'].keys())}")
        print()
    
    # Check Ollama models
    try:
        import ollama
        models = ollama.list()
        print("🤖 AVAILABLE OLLAMA MODELS:")
        for model in models['models']:
            print(f"  - {model['name']}")
        print()
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print()
    
    # Check output directories
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        midi_files = list(output_dir.glob("*.mid"))
        print(f"🎼 MIDI FILES: {len(midi_files)} files in {output_dir}")
        
        if midi_files and args.verbose:
            converter = MidiConverter()
            for midi_file in midi_files[:5]:  # Show first 5
                info = converter.get_midi_info(midi_file)
                print(f"  📄 {midi_file.name}: {info.get('length_seconds', 0):.1f}s")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PyDrums - AI-powered drum pattern generation and MIDI conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pydrums setup-data                           # Download and prepare training data
  pydrums generate -d "rock beat"              # Generate a rock pattern
  pydrums generate --interactive               # Interactive mode
  pydrums convert -p "ch: x-x-; bd: x---"      # Convert pattern to MIDI
  pydrums info                                 # Show system information
        """
    )
    
    # Global arguments
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='midi_output', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup data command
    setup_parser = subparsers.add_parser('setup-data', help='Setup training data')
    setup_parser.add_argument('--force-reload', action='store_true', 
                             help='Force re-download of data')
    setup_parser.add_argument('--skip-primary', action='store_true',
                             help='Skip primary stephenhandley source')
    setup_parser.add_argument('--skip-260', action='store_true',
                             help='Skip DrumMachinePatterns260 source')
    setup_parser.add_argument('--additional-url', help='Additional JSON data source URL')
    setup_parser.add_argument('--additional-name', help='Name for additional data source')
    
    # Regenerate training data command (NEW)
    regen_parser = subparsers.add_parser('regenerate-training', 
                                        help='Regenerate training data with speed variations')
    regen_parser.add_argument('--data-dir', default='data', help='Data directory')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate drum patterns')
    gen_parser.add_argument('-d', '--description', help='Pattern description')
    gen_parser.add_argument('-i', '--interactive', action='store_true',
                           help='Interactive mode')
    gen_parser.add_argument('-b', '--batch-file', help='File with pattern descriptions')
    gen_parser.add_argument('-m', '--model', default='llama3.1:latest',
                           help='Ollama model to use')
    gen_parser.add_argument('-e', '--examples', type=int, default=3,
                           help='Number of examples for few-shot learning')
    gen_parser.add_argument('-t', '--temperature', type=float, default=0.7,
                           help='Generation temperature')
    gen_parser.add_argument('--to-midi', action='store_true',
                           help='Convert generated patterns to MIDI')
    gen_parser.add_argument('--save-patterns', action='store_true',
                           help='Save generated patterns to JSON')
    gen_parser.add_argument('--tempo', type=int, default=120,
                           help='MIDI tempo (BPM)')
    gen_parser.add_argument('--include-tempo', action='store_true',
                           help='Include tempo information in MIDI files (default: tempo-neutral)')
    gen_parser.add_argument('--loops', type=int, default=4,
                           help='Number of pattern loops in MIDI')
    
    # Convert command
    conv_parser = subparsers.add_parser('convert', help='Convert patterns to MIDI')
    conv_parser.add_argument('-p', '--pattern', help='Pattern string to convert')
    conv_parser.add_argument('-f', '--patterns-file', help='JSON file with patterns')
    conv_parser.add_argument('-d', '--description', help='Pattern description')
    conv_parser.add_argument('--tempo', type=int, default=120, help='MIDI tempo (BPM)')
    conv_parser.add_argument('--include-tempo', action='store_true',
                            help='Include tempo information in MIDI files (default: tempo-neutral)')
    conv_parser.add_argument('--loops', type=int, default=4, help='Number of loops')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'setup-data':
            setup_data_command(args)
        elif args.command == 'regenerate-training':
            regenerate_training_data_command(args)
        elif args.command == 'generate':
            generate_command(args)
        elif args.command == 'convert':
            convert_command(args)
        elif args.command == 'info':
            info_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
