#!/usr/bin/env python3
"""
Compare embedding-based vs keyword-based example selection
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pydrums.pattern_generator import PatternGenerator

def compare_search_methods():
    """Compare embedding vs keyword search with challenging queries"""
    
    print("🔍 COMPARING SEARCH METHODS: Embeddings vs Keywords")
    print("=" * 70)
    
    # Test queries that should benefit from semantic understanding
    test_queries = [
        "Create a groovy disco beat",           # "groovy" + "disco" - semantic concepts
        "Make a syncopated rhythm",             # "syncopated" - musical concept
        "Generate a laid-back groove",          # "laid-back" - feeling/style
        "Create a driving rock pattern",        # "driving" - intensity concept
        "Make a bouncy funk beat",              # "bouncy" - rhythmic feel
        "Generate a smooth jazz rhythm",        # "smooth" - texture concept
        "Create an energetic dance beat",       # "energetic" - energy concept
        "Make a complex polyrhythmic pattern",  # "polyrhythmic" - advanced concept
    ]
    
    # Initialize generator
    generator = PatternGenerator()
    
    if not (generator.training_embeddings is not None and generator.embedding_model is not None):
        print("❌ Embeddings not available. Run: pydrums regenerate-training")
        return
    
    print(f"📊 Testing with {len(generator.training_data)} training examples")
    print(f"🧠 Embedding model: {type(generator.embedding_model).__name__}")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 60)
        
        # Test embedding-based search
        print("🎯 SEMANTIC SEARCH (Embeddings):")
        emb_examples, emb_speed, emb_fallback = generator._find_relevant_examples_with_embeddings(
            query, None, 3
        )
        
        print(f"   Examples found: {[ex.get('input', '')[:40] + '...' for ex in emb_examples]}")
        print(f"   Speed detected: {emb_speed or 'normal'}")
        
        # Test keyword-based search  
        print("🔤 KEYWORD SEARCH (Legacy):")
        kw_examples, kw_speed, kw_fallback = generator._find_relevant_examples(
            query, None, 3
        )
        
        print(f"   Examples found: {[ex.get('input', '')[:40] + '...' for ex in kw_examples]}")
        print(f"   Speed detected: {kw_speed or 'normal'}")
        
        # Compare relevance
        emb_inputs = [ex.get('input', '') for ex in emb_examples]
        kw_inputs = [ex.get('input', '') for ex in kw_examples]
        
        if emb_inputs != kw_inputs:
            print("   📈 DIFFERENT RESULTS - Semantic search found different examples!")
        else:
            print("   📊 Same results (query may have good keyword matches)")

def test_semantic_understanding():
    """Test specific semantic relationships"""
    
    print("\n\n🧠 TESTING SEMANTIC UNDERSTANDING")
    print("=" * 70)
    
    generator = PatternGenerator()
    
    # Test semantic relationships
    semantic_tests = [
        ("groovy", ["funk", "syncopated", "pocket", "rhythmic"]),
        ("driving", ["powerful", "heavy", "intense", "strong"]), 
        ("laid back", ["relaxed", "chill", "smooth", "easy"]),
        ("syncopated", ["off-beat", "complex", "funk", "groove"]),
        ("bouncy", ["lively", "energetic", "upbeat", "rhythmic"])
    ]
    
    for query_word, related_words in semantic_tests:
        print(f"\n🎯 Testing: '{query_word}' should relate to {related_words}")
        
        query = f"Create a {query_word} beat"
        examples, _, _ = generator._find_relevant_examples_with_embeddings(query, None, 5)
        
        # Check if examples contain semantically related words
        found_relations = []
        for example in examples:
            example_text = example.get('input', '').lower()
            for word in related_words:
                if word in example_text:
                    found_relations.append(word)
        
        unique_relations = list(set(found_relations))
        
        if unique_relations:
            print(f"   ✅ Found semantic relations: {unique_relations}")
        else:
            print(f"   ⚠️  No obvious semantic relations found")
            print(f"   Top example: '{examples[0].get('input', '') if examples else 'None'}'")

if __name__ == "__main__":
    compare_search_methods()
    test_semantic_understanding()