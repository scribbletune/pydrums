"""
AI-powered drum pattern generator
Uses Ollama and few-shot learning to generate drum patterns from text descriptions
"""

import ollama
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class PatternGenerator:
    """Generate drum patterns using AI and few-shot learning"""
    
    def __init__(self, model_name: str = "llama3.1:latest", data_dir: str = "data"):
        """Initialize pattern generator
        
        Args:
            model_name: Name of the Ollama model to use
            data_dir: Directory containing training data
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.training_data = []
        self._load_training_data()
        
    def _load_training_data(self):
        """Load training data for few-shot learning"""
        training_file = self.data_dir / "training_data.json"
        
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.training_data = json.load(f)
            print(f"📚 Loaded {len(self.training_data)} training examples")
        else:
            print("⚠️  No training data found. Run data collection first.")
    
    def generate_pattern(self, description: str, 
                        style_hint: Optional[str] = None,
                        num_examples: int = 3,
                        temperature: float = 0.7) -> Dict[str, Any]:
        """Generate a drum pattern from text description
        
        Args:
            description: Text description of desired pattern
            style_hint: Optional style hint to filter examples
            num_examples: Number of examples to use for few-shot learning
            temperature: Sampling temperature for generation
            
        Returns:
            Dictionary with generated pattern and metadata
        """
        if not self.training_data:
            basic_pattern = self._generate_basic_pattern(description)
            detected_speed = self._detect_speed_from_description(description.lower())
            basic_pattern['detected_speed'] = detected_speed
            basic_pattern['used_random_fallback'] = False  # Using rule-based, not random
            return basic_pattern
        
        # Find relevant examples and detect speed
        relevant_examples, detected_speed, used_random_fallback = self._find_relevant_examples(
            description, style_hint, num_examples
        )
        
        # Create prompt
        prompt = self._create_prompt(description, relevant_examples)
        
        try:
            # Generate pattern using Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': temperature,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            
            generated_text = response['message']['content'].strip()
            
            # Parse the response
            pattern_data = self._parse_generated_pattern(generated_text)
            pattern_data['description'] = description
            pattern_data['model_used'] = self.model_name
            pattern_data['examples_used'] = len(relevant_examples)
            pattern_data['detected_speed'] = detected_speed
            pattern_data['used_random_fallback'] = used_random_fallback
            
            return pattern_data
            
        except Exception as e:
            print(f"❌ Error generating pattern: {e}")
            basic_pattern = self._generate_basic_pattern(description)
            # Add detected speed to basic pattern too
            detected_speed = self._detect_speed_from_description(description.lower())
            basic_pattern['detected_speed'] = detected_speed
            basic_pattern['used_random_fallback'] = False  # Using rule-based, not random
            return basic_pattern
    
    def _find_relevant_examples(self, description: str, 
                               style_hint: Optional[str], 
                               num_examples: int) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
        """Find relevant training examples for few-shot learning, including speed detection"""
        relevant_examples = []
        keywords = description.lower().split()
        
        # Add style hint to keywords if provided
        if style_hint:
            keywords.append(style_hint.lower())
        
        # Detect speed from description using improved matching
        detected_speed = self._detect_speed_from_description(description.lower())
        
        # Also check for speed in keywords for backward compatibility
        if not detected_speed:
            speed_keywords = {
                'half-time': ['half', 'slow', 'long'],
                'double-time': ['double', 'fast', 'rapid', 'quick', 'uptempo'],
                'quarter': ['simple', 'basic', 'minimal', 'quarter']
            }
            
            for speed_type, speed_words in speed_keywords.items():
                if any(word in keywords for word in speed_words):
                    detected_speed = speed_type
                    break
        
        # Score examples by relevance
        scored_examples = []
        
        for example in self.training_data:
            score = 0
            example_text = example.get('input', '').lower()
            example_style = example.get('style', '').lower()
            example_speed = example.get('speed', 'normal')
            
            # Score based on keyword matches
            keyword_score = 0
            for keyword in keywords:
                if keyword in example_text:
                    keyword_score += 2
                if keyword in example_style:
                    keyword_score += 3
            
            score += keyword_score
            
            # Bonus score for speed matching
            if detected_speed:
                if detected_speed == 'half-time' and example_speed == 'half_time':
                    score += 5
                elif detected_speed == 'double-time' and example_speed == 'double_time':
                    score += 5
                elif detected_speed == 'quarter' and example_speed == 'quarter_notes':
                    score += 5
                # Slight penalty for mismatched speeds
                elif example_speed != 'normal' and example_speed != detected_speed.replace('-', '_'):
                    score -= 1
            else:
                # Only prefer normal speed if we have keyword matches
                if keyword_score > 0 and example_speed == 'normal':
                    score += 1
            
            if score > 0:
                scored_examples.append((score, example))
        
        # Sort by score and take top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # Check if we have no meaningful matches (all scores are 0 or no examples)
        has_meaningful_matches = len(scored_examples) > 0 and scored_examples[0][0] > 0
        used_random_fallback = not has_meaningful_matches
        
        if used_random_fallback:
            print(f"⚠️  No keyword matches found for '{description}' - using random training examples")
        
        relevant_examples = [ex[1] for ex in scored_examples[:num_examples]]
        
        # If we don't have enough relevant examples, add random ones
        if len(relevant_examples) < num_examples:
            remaining_needed = num_examples - len(relevant_examples)
            used_examples = set(id(ex) for ex in relevant_examples)
            
            available_examples = [
                ex for ex in self.training_data 
                if id(ex) not in used_examples
            ]
            
            additional = random.sample(
                available_examples, 
                min(remaining_needed, len(available_examples))
            )
            relevant_examples.extend(additional)
        
        return relevant_examples, detected_speed, used_random_fallback
    
    def _create_prompt(self, description: str, 
                      examples: List[Dict[str, Any]]) -> str:
        """Create few-shot learning prompt"""
        prompt_parts = [
            "You are a professional drum pattern generator. Create ONLY the drum pattern using this notation:",
            "",
            "NOTATION GUIDE:",
            "- x = hit/strike the drum",
            "- - = rest/silence",
            "- R = roll (extended sound)",
            "- _ = ghost note (quiet hit)",
            "- [ = flam start (grace note)",
            "- ] = flam end",
            "",
            "DRUM ABBREVIATIONS:",
            "- ch = closed hi-hat",
            "- oh = open hi-hat",
            "- sd = snare drum", 
            "- bd = bass drum",
            "- hh = hi-hat pedal",
            "- cc = crash cymbal",
            "- rc = ride cymbal",
            "- ht = high tom",
            "- mt = mid tom", 
            "- lt = low tom",
            "",
            "FORMAT: drum: pattern; drum: pattern; ...",
            "",
            "IMPORTANT: Respond with ONLY the pattern in the specified format. No explanations, no extra text.",
            "",
            "EXAMPLES:",
            ""
        ]
        
        # Add training examples
        for i, example in enumerate(examples, 1):
            prompt_parts.extend([
                f"Example {i}:",
                f"INPUT: {example.get('input', '')}",
                f"OUTPUT: {example.get('output', '')}",
                ""
            ])
        
        prompt_parts.extend([
            "Now generate a pattern for:",
            f"INPUT: {description}",
            "OUTPUT:"
        ])
        
        return "\\n".join(prompt_parts)
    
    def _parse_generated_pattern(self, generated_text: str) -> Dict[str, Any]:
        """Parse generated pattern text into structured format"""
        lines = generated_text.split('\\n')
        
        # Find the pattern line - look for drum pattern format
        pattern_lines = []
        for line in lines:
            line = line.strip()
            if ':' in line and any(drum in line.lower() for drum in ['ch', 'sd', 'bd', 'oh', 'hh']):
                # Check if it looks like a valid pattern
                parts = line.split(':', 1)
                if len(parts) == 2:
                    drum_part = parts[0].strip().lower()
                    pattern_part = parts[1].strip()
                    # Valid if drum is known and pattern has valid characters
                    if (len(drum_part) <= 4 and 
                        any(c in pattern_part for c in ['x', '-', 'R', '_', '[', ']'])):
                        pattern_lines.append(line)
        
        # Join multiple pattern lines
        if pattern_lines:
            pattern_line = "; ".join(pattern_lines)
        else:
            # Fallback to first line if no valid patterns found
            pattern_line = lines[0].strip() if lines else ""
        
        # Validate pattern
        is_valid = self._validate_pattern(pattern_line)
        
        # Extract individual drum patterns
        drum_patterns = self._extract_drum_patterns(pattern_line)
        
        return {
            'raw_output': generated_text,
            'pattern_line': pattern_line,
            'drum_patterns': drum_patterns,
            'is_valid': is_valid,
            'validation_notes': self._get_validation_notes(pattern_line)
        }
    
    def _validate_pattern(self, pattern: str) -> bool:
        """Validate if pattern follows correct format"""
        if not pattern:
            return False
        
        # Check for basic structure
        has_drum_notation = any(drum in pattern.lower() for drum in ['ch:', 'sd:', 'bd:', 'oh:', 'hh:'])
        has_pattern_chars = any(char in pattern for char in ['x', '-', 'R', '_', '[', ']'])
        
        return has_drum_notation and has_pattern_chars
    
    def _extract_drum_patterns(self, pattern_line: str) -> Dict[str, str]:
        """Extract individual drum patterns from pattern line"""
        drum_patterns = {}
        
        if ';' in pattern_line:
            parts = pattern_line.split(';')
        else:
            parts = [pattern_line]
        
        for part in parts:
            part = part.strip()
            if ':' in part:
                drum, pattern = part.split(':', 1)
                drum = drum.strip().lower()
                pattern = pattern.strip()
                
                # Clean drum name
                drum_clean = ''.join(c for c in drum if c.isalpha())
                if drum_clean:
                    drum_patterns[drum_clean] = pattern
        
        return drum_patterns
    
    def _get_validation_notes(self, pattern: str) -> List[str]:
        """Get validation notes for pattern"""
        notes = []
        
        if not pattern:
            notes.append("Empty pattern")
            return notes
        
        # Check for common issues
        if not any(drum in pattern.lower() for drum in ['ch', 'sd', 'bd']):
            notes.append("Missing common drums (ch, sd, bd)")
        
        if not any(char in pattern for char in ['x', '-']):
            notes.append("Missing basic notation (x, -)")
        
        if len(pattern) < 10:
            notes.append("Pattern seems very short")
        
        return notes
    
    def _generate_basic_pattern(self, description: str) -> Dict[str, Any]:
        """Generate a basic pattern when no training data is available"""
        # Simple rule-based generation as fallback
        basic_patterns = {
            'rock': "ch: x---x---x---x---; sd: ----x-------x---; bd: x-----x-x-------",
            'funk': "ch: x-x-x-x-x-x-x-x-; sd: ----x--x----x---; bd: x-----x---x-----",
            'jazz': "ch: x-x-x-x-x-x-x-x-; sd: ----x-------x-x-; bd: x---------x-----",
            'pop': "ch: x-x-x-x-x-x-x-x-; sd: ----x-------x---; bd: x-------x-------"
        }
        
        # Try to match description to a style
        desc_lower = description.lower()
        for style, pattern in basic_patterns.items():
            if style in desc_lower:
                return {
                    'pattern_line': pattern,
                    'drum_patterns': self._extract_drum_patterns(pattern),
                    'is_valid': True,
                    'description': description,
                    'model_used': 'rule-based-fallback',
                    'validation_notes': ['Generated using rule-based fallback']
                }
        
        # Default rock pattern
        default_pattern = basic_patterns['rock']
        return {
            'pattern_line': default_pattern,
            'drum_patterns': self._extract_drum_patterns(default_pattern),
            'is_valid': True,
            'description': description,
            'model_used': 'rule-based-fallback',
            'validation_notes': ['Generated using default rock pattern']
        }
    
    def _detect_speed_from_description(self, description_lower: str) -> Optional[str]:
        """Detect speed from description using robust pattern matching"""
        # Normalize the description for better matching
        normalized = description_lower.replace('-', ' ').replace('_', ' ')
        
        # Speed patterns with multiple variations
        speed_patterns = {
            'half-time': [
                'half time', 'halftime', 'half-time',
                'slow', 'slower', 'laid back', 'laidback', 'laid-back',
                'long notes', 'stretched', 'relaxed'
            ],
            'double-time': [
                'double time', 'doubletime', 'double-time',
                'fast', 'faster', 'rapid', 'quick', 'quickly',
                'uptempo', 'up tempo', 'high tempo',
                'short notes', 'compressed'
            ],
            'quarter': [
                'quarter notes', 'quarter note', 'simple', 'basic',
                'minimal', 'stripped down', 'four on the floor',
                'strong beats', 'downbeats'
            ]
        }
        
        # Check each speed pattern
        for speed_type, patterns in speed_patterns.items():
            for pattern in patterns:
                if pattern in normalized:
                    return speed_type
        
        return None
    
    def batch_generate(self, descriptions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate multiple patterns from a list of descriptions
        
        Args:
            descriptions: List of pattern descriptions
            **kwargs: Additional arguments for generate_pattern
            
        Returns:
            List of generated pattern dictionaries
        """
        results = []
        
        for i, description in enumerate(descriptions, 1):
            print(f"🎵 Generating {i}/{len(descriptions)}: {description}")
            
            result = self.generate_pattern(description, **kwargs)
            results.append(result)
        
        return results
    
    def interactive_mode(self):
        """Start interactive pattern generation mode"""
        print("🎮 INTERACTIVE PATTERN GENERATOR")
        print("=" * 50)
        print("Enter text descriptions to generate drum patterns!")
        print("Type 'quit' to exit")
        print()
        
        while True:
            user_input = input("🎵 Describe a drum pattern: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            result = self.generate_pattern(user_input)
            
            print(f"🥁 PATTERN: {result['pattern_line']}")
            print(f"✅ Valid: {result['is_valid']}")
            
            detected_speed = result.get('detected_speed')
            if detected_speed and detected_speed != 'normal':
                print(f"🎯 Detected Speed: {detected_speed}")
            else:
                print(f"🎯 Speed: normal (16th notes)")
            
            # Show if random examples were used
            if result.get('used_random_fallback', False):
                print(f"⚠️  Used random examples (no keyword matches found)")
            
            if result['validation_notes']:
                print(f"📝 Notes: {', '.join(result['validation_notes'])}")
            
            print()
