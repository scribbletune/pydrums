#!/usr/bin/env python3
"""
Test script to load and preview the DrumMachinePatterns260 data source
"""

from pydrums import DataLoader
import json

def main():
    print("🧪 Testing DrumMachinePatterns260 Data Source")
    print("=" * 60)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load the DrumMachinePatterns260 source
    print("📥 Loading DrumMachinePatterns260...")
    patterns_260 = loader.load_drum_machine_patterns_260(force_reload=True)
    
    if not patterns_260:
        print("❌ Failed to load patterns")
        return
    
    print(f"✅ Successfully loaded {len(patterns_260)} patterns")
    print()
    
    # Show some sample patterns
    print("📋 Sample Patterns:")
    print("-" * 40)
    
    for i, pattern in enumerate(patterns_260[:5]):
        print(f"\\n{i+1}. Pattern Info:")
        print(f"   Name: {pattern.get('name', 'Unknown')}")
        print(f"   Category: {pattern.get('category', 'Unknown')}")
        print(f"   Source: {pattern.get('source', 'Unknown')}")
        
        # Show available keys to understand the structure
        print(f"   Available keys: {list(pattern.keys())}")
        
        # If there are drums, show a sample
        if 'drums' in pattern:
            drums = pattern['drums']
            print(f"   Drums available: {list(drums.keys()) if isinstance(drums, dict) else 'Not a dict'}")
            
            if isinstance(drums, dict) and drums:
                # Show first drum pattern as example
                first_drum = list(drums.keys())[0]
                first_pattern = drums[first_drum]
                print(f"   Sample ({first_drum}): {first_pattern[:20] if isinstance(first_pattern, str) else first_pattern}")
    
    # Try converting to training data
    print("\\n🔄 Converting to training data...")
    training_data = loader.convert_patterns_to_training_data(patterns_260)
    
    if training_data:
        print(f"✅ Generated {len(training_data)} training examples")
        
        # Show sample training data
        print("\\n📚 Sample Training Examples:")
        print("-" * 40)
        
        for i, example in enumerate(training_data[:3]):
            print(f"\\n{i+1}. Training Example:")
            print(f"   Input: {example.get('input', 'Unknown')}")
            print(f"   Output: {example.get('output', 'Unknown')[:60]}...")
            print(f"   Style: {example.get('style', 'Unknown')}")
    else:
        print("❌ Failed to generate training data")
    
    # Get statistics
    stats = loader.get_data_statistics(training_data)
    
    print("\\n📊 STATISTICS:")
    print("-" * 20)
    print(f"Total examples: {stats.get('total_examples', 0)}")
    print(f"Unique styles: {stats.get('unique_styles', 0)}")
    
    if 'style_distribution' in stats:
        print("\\nStyle distribution:")
        for style, count in list(stats['style_distribution'].items())[:10]:
            print(f"  {style}: {count}")
    
    print("\\n🎉 Test completed!")

if __name__ == "__main__":
    main()
