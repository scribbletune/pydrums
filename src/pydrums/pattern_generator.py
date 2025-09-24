"""
AI-powered drum pattern generator
Uses Ollama and few-shot learning to generate drum patterns from text descriptions
"""

import ollama
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
        self.training_embeddings = None
        self.embedding_model = None
        self._load_training_data()
        self._load_embedding_model()
        
    def _load_training_data(self):
        """Load training data for few-shot learning"""
        training_file = self.data_dir / "training_data.json"
        
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.training_data = json.load(f)
            print(f"📚 Loaded {len(self.training_data)} training examples")
            
            # Try to load precomputed embeddings
            self._load_embeddings()
        else:
            print("⚠️  No training data found. Run data collection first.")
    
    def _load_embeddings(self):
        """Load precomputed embeddings"""
        from .data_loader import DataLoader
        
        loader = DataLoader(self.data_dir)
        self.training_embeddings = loader.load_embeddings()
        
        if self.training_embeddings is not None:
            print(f"🎯 Semantic search enabled with embeddings")
        else:
            print(f"⚠️  No embeddings found. Using keyword matching fallback")
    
    def _load_embedding_model(self):
        """Load embedding model for query encoding"""
        if self.training_embeddings is not None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"🧠 Embedding model loaded for query processing")
            except ImportError:
                print("⚠️  sentence-transformers not installed. Using keyword fallback.")
                self.training_embeddings = None
            except Exception as e:
                print(f"❌ Error loading embedding model: {e}")
                self.training_embeddings = None
    
    def generate_pattern(self, description: str, 
                        style_hint: Optional[str] = None,
                        num_examples: int = 3,
                        temperature: float = 0.7,
                        bars: int = 1,
                        add_fill: bool = None) -> Dict[str, Any]:
        """Generate a drum pattern from text description
        
        Args:
            description: Text description of desired pattern
            style_hint: Optional style hint to filter examples
            num_examples: Number of examples to use for few-shot learning
            temperature: Sampling temperature for generation
            bars: Number of bars to generate (1-8)
            add_fill: Whether to add drum fill in last bar (auto-detect if None)
            
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
        if self.training_embeddings is not None and self.embedding_model is not None:
            relevant_examples, detected_speed, used_random_fallback = self._find_relevant_examples_with_embeddings(
                description, style_hint, num_examples
            )
        else:
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
            
            # Extend to multiple bars and add fill if requested
            if bars > 1 or add_fill:
                pattern_data = self._extend_pattern_with_fill(pattern_data, bars, add_fill, description)
            
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
    
    def _find_relevant_examples_with_embeddings(self, description: str, 
                                              style_hint: Optional[str], 
                                              num_examples: int) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
        """Find relevant training examples using semantic similarity with embeddings"""
        
        # Detect speed from description
        detected_speed = self._detect_speed_from_description(description.lower())
        
        # Encode user query
        query_embedding = self.embedding_model.encode([description])
        
        # Compute semantic similarities
        similarities = cosine_similarity(query_embedding, self.training_embeddings)[0]
        
        # Score examples with hybrid approach
        scored_examples = []
        
        for i, example in enumerate(self.training_data):
            # Base semantic similarity score (0-1) scaled to match keyword scoring
            semantic_score = similarities[i] * 10
            
            # Speed matching bonus
            example_speed = example.get('speed', 'normal')
            if detected_speed:
                if detected_speed == 'half-time' and example_speed == 'half_time':
                    semantic_score += 5
                elif detected_speed == 'double-time' and example_speed == 'double_time':
                    semantic_score += 5
                elif detected_speed == 'quarter' and example_speed == 'quarter_notes':
                    semantic_score += 5
                elif example_speed != 'normal' and example_speed != detected_speed.replace('-', '_'):
                    semantic_score -= 1
            
            # Style hint bonus
            if style_hint:
                example_style = example.get('style', '').lower()
                if style_hint.lower() in example_style:
                    semantic_score += 3
            
            scored_examples.append((semantic_score, example))
        
        # Sort by similarity and take top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # Check if we have meaningful semantic matches
        top_score = scored_examples[0][0] if scored_examples else 0
        semantic_threshold = 3.0  # Minimum semantic similarity threshold
        has_meaningful_matches = top_score >= semantic_threshold
        used_random_fallback = not has_meaningful_matches
        
        if used_random_fallback:
            print(f"⚠️  Low semantic similarity for '{description}' - using top matches anyway")
        else:
            print(f"🎯 Found {len([s for s, _ in scored_examples if s >= semantic_threshold])} semantically similar examples")
        
        relevant_examples = [ex[1] for ex in scored_examples[:num_examples]]
        
        return relevant_examples, detected_speed, used_random_fallback
    
    def _create_prompt(self, description: str, 
                      examples: List[Dict[str, Any]]) -> str:
        """Create few-shot learning prompt"""
        prompt_parts = [
            "You are a professional drum pattern generator. Create ONLY the drum pattern using this notation:",
            "",
            "NOTATION GUIDE:",
            "- x = hit/strike the drum (accent)",
            "- X = loud hit (forte)",
            "- o = medium hit",
            "- _ = ghost note (quiet hit/pianissimo)",
            "- - = rest/silence",
            "- R = roll (extended sound)",
            "- r = short roll/buzz",
            "- [ = flam start (grace note)",
            "- ] = flam end",
            "- ^ = accent/emphasis",
            "- . = staccato/short",
            "",
            "DRUM ABBREVIATIONS:",
            "- bd = bass drum/kick",
            "- sd = snare drum", 
            "- ch = closed hi-hat",
            "- oh = open hi-hat",
            "- hh = hi-hat pedal",
            "- rc = ride cymbal",
            "- cc = crash cymbal",
            "- rs = rim shot/side stick",
            "- ht = high tom",
            "- mt = mid tom", 
            "- lt = low tom",
            "- cy = cymbal",
            "- tb = tambourine",
            "- cb = cowbell",
            "- cp = clap",
            "",
            "FORMAT: drum: pattern; drum: pattern; ...",
            "",
            "CRITICAL RULES:",
            "1. Each pattern should be 8-16 characters long for proper groove",
            "2. Maintain rhythm throughout - avoid long stretches of silence (----)",
            "3. Create complete musical phrases that loop naturally",
            "4. Use varied dynamics (X, x, o, _, -) for musicality",
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
        """Parse generated pattern text into structured format with normalization"""
        lines = generated_text.split('\\n')
        
        # Find the pattern line - look for drum pattern format
        pattern_lines = []
        for line in lines:
            line = line.strip()
            if ':' in line and any(drum in line.lower() for drum in ['ch', 'sd', 'bd', 'oh', 'hh', 'rc']):
                # Check if it looks like a valid pattern
                parts = line.split(':', 1)
                if len(parts) == 2:
                    drum_part = parts[0].strip().lower()
                    pattern_part = parts[1].strip()
                    # Valid if drum is known and pattern has valid characters
                    if (len(drum_part) <= 4 and 
                        any(c in pattern_part for c in ['x', 'X', 'o', '_', '-', 'R', 'r', '[', ']', '^', '.'])):
                        pattern_lines.append(line)
        
        # Join multiple pattern lines
        if pattern_lines:
            pattern_line = "; ".join(pattern_lines)
        else:
            # Fallback to first line if no valid patterns found
            pattern_line = lines[0].strip() if lines else ""
        
        # Clean and normalize the pattern
        pattern_line = self._clean_and_normalize_pattern(pattern_line)
        
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
        has_drum_notation = any(drum in pattern.lower() for drum in ['ch:', 'sd:', 'bd:', 'oh:', 'hh:', 'rc:'])
        has_pattern_chars = any(char in pattern for char in ['x', 'X', 'o', '_', '-', 'R', 'r', '[', ']', '^', '.'])
        
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
        
        if not any(char in pattern for char in ['x', 'X', 'o', '_', '-']):
            notes.append("Missing basic notation (x, X, o, _, -)")
        
        if len(pattern) < 10:
            notes.append("Pattern seems very short")
        
        return notes
    
    def _clean_and_normalize_pattern(self, pattern_line: str) -> str:
        """Clean pattern line and normalize all drum patterns to 16 beats"""
        if not pattern_line:
            return pattern_line
        
        # Remove extra whitespace and clean formatting
        pattern_line = ' '.join(pattern_line.split())
        
        # Split into drum parts
        parts = pattern_line.split(';')
        normalized_parts = []
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip().lower()
            pattern = pattern.strip()
            
            # Remove any spaces within the pattern
            pattern = ''.join(pattern.split())
            
            # Normalize pattern length to 16 beats - REPEAT pattern instead of padding with silence
            if len(pattern) > 16:
                pattern = pattern[:16]  # Truncate if too long
            elif len(pattern) < 16:
                # Repeat the pattern to fill 16 beats instead of padding with silence
                if len(pattern) > 0:
                    repetitions = 16 // len(pattern)
                    remainder = 16 % len(pattern)
                    pattern = (pattern * repetitions) + pattern[:remainder]
                else:
                    pattern = '-' * 16  # Fallback for empty patterns
            
            # Clean drum name
            drum_clean = ''.join(c for c in drum if c.isalpha())
            if drum_clean:
                normalized_parts.append(f"{drum_clean}: {pattern}")
        
        return "; ".join(normalized_parts)
    
    def _extend_pattern_with_fill(self, pattern_data: Dict[str, Any], bars: int, add_fill: bool, description: str) -> Dict[str, Any]:
        """Extend pattern to multiple bars and optionally add drum fill"""
        if bars < 1:
            bars = 1
        if bars > 8:
            bars = 8  # Reasonable limit
            
        # Auto-detect if fill should be added
        if add_fill is None:
            add_fill = self._should_add_fill(description, bars)
        
        base_pattern = pattern_data['pattern_line']
        drum_patterns = pattern_data.get('drum_patterns', {})
        
        if bars == 1 and not add_fill:
            return pattern_data
        
        # Detect style for appropriate fills
        style = self._detect_style_from_description(description.lower())
        
        # Extend pattern to multiple bars
        extended_patterns = {}
        
        for drum, pattern in drum_patterns.items():
            if bars == 1:
                # Single bar - replace with fill if requested
                if add_fill:
                    fill_pattern = self._generate_drum_fill(drum, style, pattern)
                    extended_patterns[drum] = fill_pattern
                else:
                    extended_patterns[drum] = pattern
            else:
                # Multiple bars - repeat ORIGINAL pattern for all but last bar
                if add_fill:
                    # Use original pattern for first (bars-1) bars, then fill for last bar
                    repeated_pattern = pattern * (bars - 1)
                    fill_pattern = self._generate_drum_fill(drum, style, pattern)
                    extended_patterns[drum] = repeated_pattern + fill_pattern
                else:
                    # No fill - just repeat original pattern for all bars
                    extended_patterns[drum] = pattern * bars
        
        # Reconstruct pattern line
        extended_parts = []
        for drum, pattern in extended_patterns.items():
            extended_parts.append(f"{drum}: {pattern}")
        
        # Update pattern data
        pattern_data['pattern_line'] = "; ".join(extended_parts)
        pattern_data['drum_patterns'] = extended_patterns
        pattern_data['bars'] = bars
        pattern_data['has_fill'] = add_fill
        pattern_data['total_length'] = len(next(iter(extended_patterns.values()))) if extended_patterns else 16
        
        return pattern_data
    
    def _should_add_fill(self, description: str, bars: int) -> bool:
        """Auto-detect if drum fill should be added based on description and bar count"""
        desc_lower = description.lower()
        
        # Always add fill for multi-bar patterns unless explicitly avoided
        if bars >= 4:
            return True
        elif bars >= 2:
            # Add fill for 2-3 bars if description suggests it
            fill_keywords = ['fill', 'break', 'outro', 'ending', 'transition', 'buildup']
            return any(keyword in desc_lower for keyword in fill_keywords)
        else:
            # Single bar - only add if explicitly requested
            fill_keywords = ['fill', 'break', 'roll', 'buildup']
            return any(keyword in desc_lower for keyword in fill_keywords)
    
    def _detect_style_from_description(self, description: str) -> str:
        """Detect musical style from description for appropriate fills"""
        style_keywords = {
            'rock': ['rock', 'metal', 'punk', 'grunge', 'alternative'],
            'funk': ['funk', 'funky', 'groove', 'syncopated'],
            'jazz': ['jazz', 'swing', 'bebop', 'smooth', 'brushes'],
            'latin': ['latin', 'salsa', 'samba', 'bossa', 'mambo'],
            'reggae': ['reggae', 'ska', 'one drop', 'jamaica'],
            'blues': ['blues', 'shuffle', 'slow', 'soulful'],
            'pop': ['pop', 'commercial', 'radio', 'catchy'],
            'disco': ['disco', 'four on the floor', 'dance']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in description for keyword in keywords):
                return style
        
        return 'rock'  # Default style
    
    def _generate_drum_fill(self, drum: str, style: str, base_pattern: str) -> str:
        """Generate a 16-beat drum fill pattern for the specified drum and style"""
        
        # Style-specific fill patterns (16 beats each) - More dramatic and obvious
        fill_patterns = {
            'rock': {
                'bd': 'x-x-x-x-x-x-x-X-',  # Fast kick building to accent
                'sd': '----x---x-r-R-R-',  # Normal hits then rolls at END
                'ch': 'x-x-x-x---------',   # Hi-hat for first half, space for fill
                'oh': '--------x---x---',   # Open hats accent the fill
                'ht': '------x---x-x-x-',   # High tom buildup with more activity
                'mt': '--------x---x-x-',   # Mid tom cascade with more hits
                'lt': '----------x-x-X-',   # Low tom finale building up
                'rc': '----------------',   # Drop out completely
                'cc': '---------------X'    # Big crash ending
            },
            'funk': {
                'bd': 'x---x---x-X-x-X-',  # Syncopated with accents
                'sd': '----x-_-x-r-R-R-',  # Ghost notes + normal hits + rolls at END
                'ch': 'x-x-x-x-x-x-----',   # Hi-hat groove then space
                'oh': '----------------',   # Drop out
                'ht': '--------x---x-x-',   # Tom buildup at end with more activity
                'mt': '----------x-x-x-',   # Mid tom cascade with more hits
                'lt': '------------x-X-',   # Low tom ending with buildup
                'rc': '----------------'    # Drop out
            },
            'jazz': {
                'bd': 'x-------x-------',  # Sparse kick (jazz style)
                'sd': 'x-x---x---r-r-r-',  # Light snare hits then brush rolls at end
                'ch': 'x-x-x-x---------',   # Hi-hat for first half
                'oh': '--------x---x---',   # Open hats for accents
                'rc': 'x-x-x-x-r-r-R-R-',  # Ride pattern then roll builds at end
                'ht': '--------x---x-x-',   # Tom buildup at end with more activity
                'mt': '----------x-x-x-',   # Mid tom cascade with more hits
                'lt': '------------x-x-'    # Tom ending with buildup
            },
            'latin': {
                'bd': 'x---x---x-x-x-X-',  # Building Latin kick
                'sd': '----x-x---r-R-R-',  # Latin snare pattern then rolls at END
                'ch': 'x-x-x-x---------',   # Hi-hat pattern then space
                'rs': 'x-x-x-x-x-x-x-x-',  # Continuous rim shots (Latin style)
                'ht': '------x---x-x-x-',   # High tom buildup with more activity
                'mt': '--------x-x-x-x-',   # Mid tom cascade with more hits
                'lt': '----------x-x-X-',   # Low tom finale building up
                'cb': 'x-x-x-x-x-x-x-X-'    # Cowbell continuous with final accent
            }
        }
        
        # Get style-specific patterns, fall back to rock
        style_fills = fill_patterns.get(style, fill_patterns['rock'])
        
        # Get fill pattern for this drum, or create a generic one
        if drum in style_fills:
            return style_fills[drum]
        else:
            # Generic dramatic fills based on drum type - less empty space
            if drum in ['bd']:
                return 'x-x-x-x-x-x-x-X-'  # Fast kick building to accent
            elif drum in ['sd']:
                return '----x---x-r-R-R-'  # Normal hits then roll BUILDS at the end
            elif drum in ['ch']:
                return 'x-x-x-x---------'   # Hi-hat groove then space for fill
            elif drum in ['oh']:
                return '--------x---x---'   # Open hats accent the fill
            elif drum in ['ht']:
                return '------x---x-x-x-'   # High tom buildup with more activity
            elif drum in ['mt']:
                return '--------x---x-x-'   # Mid tom cascade with more hits
            elif drum in ['lt']:
                return '----------x-x-X-'   # Low tom finale building up
            elif drum in ['cc']:
                return '---------------X'    # Big crash ending
            elif drum in ['rc']:
                return 'x-x---x-r-r-R-R-'   # Ride pattern then roll builds at end
            elif drum in ['rs']:
                return 'x-x-x-x-x-x-x-x-'   # Continuous rim shots
            else:
                # For other drums, create a less empty buildup pattern
                return '----x---x-x-x-x-'   # Generic buildup with more activity
    
    def _generate_basic_pattern(self, description: str) -> Dict[str, Any]:
        """Generate a basic pattern when no training data is available"""
        # Enhanced rule-based generation with musical dynamics
        basic_patterns = {
            'rock': "bd: x---x---x---x---; sd: ----X-------X---; ch: x-x-x-x-x-x-x-x-",
            'funk': "bd: x-----x---x-----; sd: ----X--_----X-_-; ch: x-x-x-x-x-x-x-x-; rc: ----o-o-----o-o-",
            'jazz': "bd: x---------x-----; sd: ----x-------x-_-; rc: x-o-x-o-x-o-x-o-; hh: ----x---x---x---",
            'pop': "bd: x-------x-------; sd: ----x-------x---; ch: x-x-x-x-x-x-x-x-",
            'disco': "bd: x---x---x---x---; sd: ----x-------x---; ch: x-x-x-x-x-x-x-x-; oh: ------x-------x-",
            'reggae': "bd: ----x-------x---; sd: ----X-------X---; ch: x-x---x-x-x---x-",
            'latin': "bd: x-------x-x-----; sd: ----x--_----x---; rs: --x---x---x---x-",
            'afro': "bd: x-------x-----x-; sd: ----x--_----x-_-; ch: x-x-x-x-x-x-x-x-; rs: --x-----x------"
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
    
    def interactive_mode(self, midi_converter=None, **midi_kwargs):
        """Start interactive pattern generation mode with MIDI conversion"""
        print("🎮 INTERACTIVE PATTERN GENERATOR")
        print("=" * 50)
        print("Enter text descriptions to generate drum patterns!")
        print("Type 'quit' to exit")
        print()
        
        # Import here to avoid circular imports
        if midi_converter is None:
            from .midi_converter import MidiConverter
            midi_converter = MidiConverter()
        
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
            
            # Convert to MIDI automatically
            if result['is_valid']:
                try:
                    midi_path = midi_converter.pattern_to_midi(result, **midi_kwargs)
                    print(f"💾 MIDI saved: {midi_path}")
                except Exception as e:
                    print(f"❌ MIDI conversion failed: {e}")
            
            print()
