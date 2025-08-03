#!/usr/bin/env python3
"""
Regenerate improved training data with better musical diversity and accuracy
"""

from pydrums import DataLoader
import json
import os
from pathlib import Path

def main():
    print("🔄 REGENERATING IMPROVED TRAINING DATA")
    print("=" * 60)
    
    # Backup existing training data
    data_dir = Path("data")
    existing_training = data_dir / "training_data.json"
    
    if existing_training.exists():
        backup_path = data_dir / "training_data_backup.json"
        print(f"📦 Backing up existing training data to {backup_path}")
        os.rename(existing_training, backup_path)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load patterns from the new converted file
    print("📥 Loading drum patterns from converted source with improved format...")
    patterns = loader.load_drum_machine_patterns_260(use_converted=True)
    
    if not patterns:
        print("❌ Failed to load patterns from converted file")
        print("Trying fallback to original source...")
        patterns = loader.load_drum_machine_patterns_260(use_converted=False)
        
        if not patterns:
            print("❌ Failed to load patterns from any source")
            return
    
    print(f"✅ Loaded {len(patterns)} patterns with enhanced format")
    
    # Convert to improved training data
    print("🎵 Converting to enhanced training data...")
    training_data = loader.convert_patterns_to_training_data(patterns)
    
    print(f"✅ Generated {len(training_data)} training examples")
    
    # Save the new training data
    loader.save_training_data(training_data, "training_data.json")
    
    # Get statistics
    stats = loader.get_data_statistics(training_data)
    
    print("\n📊 NEW TRAINING DATA STATISTICS:")
    print("-" * 40)
    print(f"Total examples: {stats.get('total_examples', 0)}")
    print(f"Unique styles: {stats.get('unique_styles', 0)}")
    print(f"Average pattern length: {stats.get('avg_pattern_length', 0):.1f} characters")
    
    if 'style_distribution' in stats:
        print("\nStyle distribution:")
        for style, count in stats['style_distribution'].items():
            print(f"  {style}: {count} examples")
    
    if 'time_signatures' in stats:
        print("\nTime signatures:")
        for sig, count in stats['time_signatures'].items():
            print(f"  {sig}: {count} patterns")
    
    # Show some sample improved examples
    print("\n🎼 SAMPLE IMPROVED PATTERNS:")
    print("-" * 40)
    
    # Show examples from different styles
    styles_shown = set()
    for example in training_data[:20]:
        style = example.get('style', 'unknown')
        if style not in styles_shown and len(styles_shown) < 5:
            styles_shown.add(style)
            print(f"\n{style.upper()}:")
            print(f"  Input: {example.get('input', '')}")
            print(f"  Output: {example.get('output', '')}")
            print(f"  Time: {example.get('time_signature', 'unknown')}")
            if 'speed' in example and example['speed']:
                print(f"  Speed: {example['speed']}")
    
    print("\n🎉 Training data regeneration complete!")
    print(f"📁 New training data saved to: {data_dir / 'training_data.json'}")
    print("🎵 Your patterns now include:")
    print("   • Enhanced drum notation with dynamics (X, o, _, etc.)")
    print("   • More diverse musical descriptions")
    print("   • Better style differentiation")
    print("   • Speed variations (half-time, double-time, quarter notes)")
    print("   • Reduced redundancy in training examples")

if __name__ == "__main__":
    main()