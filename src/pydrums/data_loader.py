"""
Data loader for drum pattern datasets
Handles loading and processing of training data from various sources
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import pickle
import numpy as np


class DataLoader:
    """Load and process drum pattern data from various sources"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data loader
        
        Args:
            data_dir: Directory to store/load data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_github_json_patterns(self, 
                                 repo_url: str = "https://api.github.com/repos/stephenhandley/DrumMachinePatterns/contents/patterns",
                                 force_reload: bool = False) -> List[Dict[str, Any]]:
        """Load drum patterns from GitHub JSON repository
        
        Args:
            repo_url: GitHub API URL for the patterns
            force_reload: Whether to re-download even if local file exists
            
        Returns:
            List of drum pattern dictionaries
        """
        cache_file = self.data_dir / "github_patterns.json"
        
        if cache_file.exists() and not force_reload:
            print(f"📂 Loading cached patterns from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        print(f"🌐 Downloading patterns from GitHub...")
        
        try:
            # Get list of pattern files
            response = requests.get(repo_url)
            response.raise_for_status()
            files = response.json()
            
            patterns = []
            
            for file_info in files:
                if file_info['name'].endswith('.json'):
                    print(f"  📄 Loading {file_info['name']}")
                    
                    # Download individual pattern file
                    file_response = requests.get(file_info['download_url'])
                    file_response.raise_for_status()
                    
                    pattern_data = file_response.json()
                    pattern_data['source_file'] = file_info['name']
                    patterns.append(pattern_data)
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(patterns, f, indent=2)
            
            print(f"✅ Loaded {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            print(f"❌ Error loading GitHub patterns: {e}")
            return []
    
    def load_additional_json_source(self, url: str, cache_name: str) -> List[Dict[str, Any]]:
        """Load patterns from an additional JSON source
        
        Args:
            url: URL to the JSON data
            cache_name: Name for the cache file
            
        Returns:
            List of pattern dictionaries
        """
        cache_file = self.data_dir / f"{cache_name}.json"
        
        if cache_file.exists():
            print(f"📂 Loading cached data from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        try:
            print(f"🌐 Downloading from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Loaded data from {url}")
            return data
            
        except Exception as e:
            print(f"❌ Error loading from {url}: {e}")
            return []
    
    def convert_patterns_to_training_data(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert raw patterns to ML training format
        
        Args:
            patterns: Raw pattern data
            
        Returns:
            List of training examples with input/output pairs
        """
        training_data = []
        
        for pattern in patterns:
            # Extract pattern information - handle different formats
            name = pattern.get('name') or pattern.get('title', 'Unknown')
            style = self._extract_style(name)
            time_signature = pattern.get('timeSignature') or pattern.get('signature', '4/4')
            
            # Handle different drum data formats
            drum_data = pattern.get('drums') or pattern.get('tracks', {})
            
            # Convert drum data to pattern notation
            pattern_str = self._convert_drums_to_notation(drum_data)
            
            # Normalize all base patterns to 16 beats for consistency
            if pattern_str:
                pattern_str = self._normalize_pattern_string(pattern_str)
            
            if pattern_str:
                # Generate multiple training examples per pattern
                training_examples = self._generate_training_examples(
                    pattern_str, style, time_signature, name
                )
                training_data.extend(training_examples)
        
        return training_data
    
    def _extract_style(self, name: str) -> str:
        """Extract musical style from pattern name"""
        name_lower = name.lower()
        
        style_mapping = {
            'rock': 'rock',
            'funk': 'funk', 
            'jazz': 'jazz',
            'latin': 'latin',
            'pop': 'pop',
            'blues': 'blues',
            'country': 'country',
            'reggae': 'reggae',
            'disco': 'disco',
            'punk': 'punk',
            'metal': 'metal',
            'hip': 'hip-hop',
            'rnb': 'r&b',
            'soul': 'soul',
            'gospel': 'gospel',
            'afro': 'afro',
            'afrocub': 'afro-cuban',
            'cuban': 'afro-cuban',
            'samba': 'latin',
            'bossa': 'bossa-nova',
            'mambo': 'latin',
            'cha': 'latin',
            'rumba': 'latin',
            'salsa': 'latin',
            'tango': 'tango',
            'swing': 'jazz',
            'shuffle': 'shuffle',
            'waltz': 'waltz',
            'march': 'march',
            'polka': 'polka',
            'beguine': 'beguine',
            'foxtrot': 'foxtrot',
            'ballad': 'ballad',
            'slow': 'ballad',
            'fast': 'uptempo',
            'medium': 'medium-tempo'
        }
        
        for key, style in style_mapping.items():
            if key in name_lower:
                return style
        
        return 'general'
    
    def _convert_drums_to_notation(self, drums: Dict[str, Any]) -> str:
        """Convert drum data to pattern notation string with enhanced musical representation"""
        if not drums:
            return ""
        
        # Check if this is the new x/- notation format (string values)
        if all(isinstance(v, str) for v in drums.values()):
            return self._convert_x_dash_notation(drums)
        
        # Enhanced drum mapping with more instruments (for legacy list format)
        drum_mapping = {
            'BassDrum': 'bd',
            'SnareDrum': 'sd', 
            'ClosedHiHat': 'ch',
            'OpenHiHat': 'oh',
            'HiHatPedal': 'hh',
            'CrashCymbal': 'cc',
            'RideCymbal': 'rc',
            'HighTom': 'ht',
            'MidTom': 'mt',
            'MediumTom': 'mt',
            'LowTom': 'lt',
            'RimShot': 'rs',
            'Cymbal': 'cy',
            'Tambourine': 'tb',
            'Cowbell': 'cb',
            'Clap': 'cp'
        }
        
        pattern_parts = []
        
        # Process drums in a musical order (rhythm section first)
        drum_order = ['BassDrum', 'SnareDrum', 'ClosedHiHat', 'OpenHiHat', 'HiHatPedal', 
                     'RideCymbal', 'CrashCymbal', 'RimShot', 'HighTom', 'MediumTom', 
                     'LowTom', 'Cymbal', 'Tambourine', 'Cowbell', 'Clap']
        
        for drum_name in drum_order:
            if drum_name in drums and drums[drum_name]:
                notation = self._hits_to_notation(drums[drum_name])
                if notation and notation != "-" * len(notation):  # Skip empty patterns
                    pattern_parts.append(f"{drum_mapping[drum_name]}: {notation}")
        
        # Add any remaining drums not in the order
        for drum_name, hits in drums.items():
            if drum_name not in drum_order and drum_name in drum_mapping and hits:
                notation = self._hits_to_notation(hits)
                if notation and notation != "-" * len(notation):
                    pattern_parts.append(f"{drum_mapping[drum_name]}: {notation}")
        
        return "; ".join(pattern_parts)
    
    def _convert_x_dash_notation(self, drums: Dict[str, str]) -> str:
        """Convert new x/- notation format to pattern string with flam support"""
        pattern_parts = []
        
        # Process drums in a musical order
        drum_order = ['bd', 'sd', 'ch', 'oh', 'hh', 'rc', 'cc', 'rs', 'ht', 'mt', 'lt', 'cy', 'tb', 'cb', 'cp']
        
        for drum_code in drum_order:
            if drum_code in drums:
                pattern = drums[drum_code]
                # Process flams and convert to enhanced notation
                enhanced_pattern = self._process_flams_and_dynamics(pattern)
                if enhanced_pattern and enhanced_pattern != "-" * len(enhanced_pattern.replace('[', '').replace(']', '')):
                    pattern_parts.append(f"{drum_code}: {enhanced_pattern}")
        
        # Add any remaining drums not in the order
        for drum_code, pattern in drums.items():
            if drum_code not in drum_order:
                enhanced_pattern = self._process_flams_and_dynamics(pattern)
                if enhanced_pattern and enhanced_pattern != "-" * len(enhanced_pattern.replace('[', '').replace(']', '')):
                    pattern_parts.append(f"{drum_code}: {enhanced_pattern}")
        
        return "; ".join(pattern_parts)
    
    def _process_flams_and_dynamics(self, pattern: str) -> str:
        """Process flams [xx] and convert x/- to enhanced notation with dynamics"""
        if not pattern:
            return pattern
        
        # Handle flams [xx] by converting to enhanced notation
        import re
        
        # Replace flams [xx] with enhanced notation
        # [xx] becomes X (accent with flam)
        pattern = re.sub(r'\[xx\]', 'F', pattern)  # F for flam
        pattern = re.sub(r'\[x\]', 'f', pattern)   # f for light flam
        
        # Convert basic x/- to enhanced notation with some dynamics
        enhanced = ""
        for i, char in enumerate(pattern):
            if char == 'x':
                # Add some musical dynamics - accents on strong beats
                if i % 4 == 0:  # Strong beats get accents
                    enhanced += 'X'
                elif i % 2 == 0:  # Medium beats get normal hits
                    enhanced += 'x'
                else:  # Weak beats get ghost notes occasionally
                    enhanced += 'o' if i % 8 == 6 else 'x'
            elif char == '-':
                enhanced += '_'  # Use underscore for cleaner rests
            else:
                enhanced += char  # Keep flams and other special characters
        
        return enhanced
    
    def _hits_to_notation(self, hits: List[str]) -> str:
        """Convert hit list to notation string"""
        notation_map = {
            'Note': 'x',
            'Rest': '-'
        }
        
        return "".join(notation_map.get(hit, '-') for hit in hits)
    
    def _generate_training_examples(self, pattern_str: str, style: str, 
                                  time_sig: str, name: str) -> List[Dict[str, str]]:
        """Generate diverse training examples from one pattern with improved musical focus"""
        examples = []
        
        # Ensure pattern doesn't get truncated by normalizing length first
        pattern_str = self._ensure_complete_pattern(pattern_str)
        
        # Create more varied and musically descriptive prompts
        style_descriptors = {
            'rock': ['heavy', 'driving', 'steady', 'powerful', 'straight', 'classic'],
            'funk': ['syncopated', 'groovy', 'tight', 'pocket', 'with ghost notes', 'bouncy'],
            'jazz': ['swinging', 'brushed', 'subtle', 'loose', 'walking', 'bebop'],
            'disco': ['four on the floor', 'danceable', 'steady', 'pumping', 'retro'],
            'reggae': ['one drop', 'laid back', 'off-beat', 'skank', 'jamaican'],
            'latin': ['syncopated', 'complex', 'polyrhythmic', 'spicy', 'afro-cuban'],
            'afro': ['polyrhythmic', 'complex', 'traditional', 'layered', 'tribal'],
            'afro-cuban': ['syncopated', 'complex', 'traditional', 'latin', 'polyrhythmic'],
            'blues': ['shuffled', 'swinging', 'laid back', 'soulful', 'twelve-bar'],
            'ballad': ['gentle', 'soft', 'slow', 'emotional', 'romantic'],
            'pop': ['catchy', 'simple', 'accessible', 'radio-friendly', 'commercial'],
            'general': ['basic', 'simple', 'standard', 'versatile', 'classic']
        }
        
        descriptors = style_descriptors.get(style, ['basic', 'simple', 'standard'])
        
        # Generate fewer but higher quality examples (2-3 base)
        base_prompts = [
            f"Create a {style} drum pattern",
            f"Generate a {descriptors[0]} {style} beat"
        ]
        
        # Add more specific prompts based on style and time signature
        if time_sig == "4/4":
            if len(descriptors) > 1:
                base_prompts.append(f"Make a {descriptors[1]} {style} groove in 4/4")
        elif time_sig == "12/8":
            base_prompts.append(f"Create a {style} shuffle in 12/8 time")
        elif time_sig == "6/8":
            base_prompts.append(f"Generate a {style} pattern in 6/8 time")
        elif time_sig == "3/4":
            base_prompts.append(f"Create a {style} waltz pattern in 3/4")
        
        # Only create 2-3 base examples for better quality
        for i, prompt in enumerate(base_prompts[:3]):
            examples.append({
                "input": prompt,
                "output": pattern_str,
                "style": style,
                "time_signature": time_sig,
                "source_pattern": name,
                "speed": "normal",
                "pattern_length": self._get_actual_pattern_length(pattern_str),
                "variation": f"base_{i+1}",
                "quality": "high"
            })
        
        # Generate fewer but better speed variations
        speed_variations = self._generate_speed_variations(pattern_str, style, time_sig, name)
        examples.extend(speed_variations[:4])  # Reduced from 6 to 4
        
        return examples
    
    def _ensure_complete_pattern(self, pattern_str: str) -> str:
        """Ensure pattern is complete and not truncated"""
        if not pattern_str:
            return pattern_str
        
        parts = pattern_str.split(';')
        fixed_parts = []
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip()
            pattern = pattern.strip()
            
            # Ensure pattern is properly terminated (not cut off mid-bar)
            # For 4/4 patterns, ensure length is divisible by 4 or extend to 16
            target_length = 16
            if len(pattern) < 16:
                # Repeat pattern to fill 16 beats
                repetitions = target_length // len(pattern)
                remainder = target_length % len(pattern)
                full_pattern = pattern * repetitions + pattern[:remainder]
                fixed_parts.append(f"{drum}: {full_pattern}")
            else:
                # Ensure it ends on a complete bar (multiple of 4)
                bar_length = 4
                complete_bars = (len(pattern) // bar_length) * bar_length
                if complete_bars > 0:
                    complete_pattern = pattern[:complete_bars]
                    fixed_parts.append(f"{drum}: {complete_pattern}")
                else:
                    fixed_parts.append(f"{drum}: {pattern}")
        
        return "; ".join(fixed_parts)
    
    def _get_actual_pattern_length(self, pattern_str: str) -> int:
        """Get the actual length of the first pattern in the string"""
        if not pattern_str or ':' not in pattern_str:
            return 16
        
        first_part = pattern_str.split(';')[0].strip()
        if ':' in first_part:
            pattern = first_part.split(':', 1)[1].strip()
            return len(pattern.replace('[', '').replace(']', ''))
        
        return 16
    
    def _generate_speed_variations(self, pattern_str: str, style: str, 
                                 time_sig: str, name: str) -> List[Dict[str, str]]:
        """Generate improved speed variation training examples with better musical accuracy"""
        variations = []
        
        # Only generate variations for styles where they make musical sense
        style_supports_halftime = style in ['rock', 'funk', 'r&b', 'hip-hop', 'reggae', 'general']
        style_supports_doubletime = style in ['punk', 'metal', 'rock', 'funk', 'jazz', 'general']
        
        # Half-time variations - only 1 example per pattern to reduce redundancy
        if style_supports_halftime:
            half_time_pattern = self._create_improved_half_time_pattern(pattern_str)
            if half_time_pattern and half_time_pattern != pattern_str:
                variations.append({
                    "input": f"Create a half-time {style} groove",
                    "output": half_time_pattern,
                    "style": style,
                    "time_signature": time_sig,
                    "source_pattern": f"{name}_halftime",
                    "speed": "half_time",
                    "pattern_length": self._get_actual_pattern_length(half_time_pattern),
                    "variation": "half_time",
                    "quality": "high"
                })
        
        # Double-time variations - only 1 example per pattern
        if style_supports_doubletime:
            double_time_pattern = self._create_improved_double_time_pattern(pattern_str)
            if double_time_pattern and double_time_pattern != pattern_str:
                variations.append({
                    "input": f"Create a double-time {style} beat",
                    "output": double_time_pattern,
                    "style": style,
                    "time_signature": time_sig,
                    "source_pattern": f"{name}_doubletime",
                    "speed": "double_time", 
                    "pattern_length": self._get_actual_pattern_length(double_time_pattern),
                    "variation": "double_time",
                    "quality": "high"
                })
        
        # Simplified variations - only for complex patterns
        if self._is_complex_pattern(pattern_str):
            simple_pattern = self._create_simplified_pattern(pattern_str)
            if simple_pattern and simple_pattern != pattern_str:
                variations.append({
                    "input": f"Create a simple {style} beat",
                    "output": simple_pattern,
                    "style": style,
                    "time_signature": time_sig,
                    "source_pattern": f"{name}_simple",
                    "speed": "normal",
                    "pattern_length": self._get_actual_pattern_length(simple_pattern),
                    "variation": "simplified",
                    "quality": "high"
                })
        
        return variations
    
    def _is_complex_pattern(self, pattern_str: str) -> bool:
        """Check if pattern is complex enough to warrant simplification"""
        if not pattern_str:
            return False
        
        # Count number of drum parts and hit density
        parts = pattern_str.split(';')
        total_hits = 0
        total_positions = 0
        
        for part in parts:
            if ':' in part:
                pattern = part.split(':', 1)[1].strip()
                hits = len([c for c in pattern if c not in ['-', '_']])
                total_hits += hits
                total_positions += len(pattern)
        
        # Complex if more than 3 drum parts or high hit density
        hit_density = total_hits / max(total_positions, 1)
        return len(parts) > 3 or hit_density > 0.4
    
    def _create_simplified_pattern(self, pattern_str: str) -> str:
        """Create a simplified version focusing on main rhythm elements"""
        if not pattern_str:
            return pattern_str
        
        parts = pattern_str.split(';')
        simplified_parts = []
        
        # Keep only essential drums: kick, snare, hi-hat
        essential_drums = ['bd', 'sd', 'ch']
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip()
            
            # Only keep essential drums
            if drum in essential_drums:
                pattern = pattern.strip()
                # Simplify the pattern - reduce density
                simplified = ""
                for i, char in enumerate(pattern):
                    if char in ['X', 'x', 'F', 'f']:
                        # Keep hits on strong beats (0, 4, 8, 12) and some off-beats
                        if i % 4 == 0 or (i % 8 == 4 and drum == 'sd'):
                            simplified += 'X' if i % 4 == 0 else 'x'
                        else:
                            simplified += '_'
                    else:
                        simplified += '_'
                
                simplified_parts.append(f"{drum}: {simplified}")
        
        return "; ".join(simplified_parts)
    
    def _create_improved_half_time_pattern(self, pattern_str: str) -> str:
        """Create improved half-time version with better musical feel"""
        if not pattern_str:
            return pattern_str
        
        parts = pattern_str.split(';')
        half_time_parts = []
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip()
            pattern = pattern.strip()
            
            # Normalize to 16 beats first
            pattern = self._normalize_pattern_length(pattern, 16)
            
            # Create half-time feel with proper musical emphasis
            half_time = ""
            for i in range(16):
                if i < len(pattern):
                    char = pattern[i]
                    # Half-time: snare on beat 3 (position 8), kick on 1 and light on 2.5
                    if drum == 'bd':  # Bass drum
                        if i in [0, 6]:  # Beat 1 and weak 2.5
                            half_time += 'X' if i == 0 else 'x'
                        else:
                            half_time += '_'
                    elif drum == 'sd':  # Snare drum
                        if i == 8:  # Beat 3 - main snare
                            half_time += 'X'
                        elif i == 14:  # Light snare on 4.5
                            half_time += 'o'
                        else:
                            half_time += '_'
                    elif drum in ['ch', 'hh']:  # Hi-hat
                        if i % 2 == 0:  # Keep steady 8th notes but lighter
                            half_time += 'o' if char in ['x', 'X', 'o'] else '_'
                        else:
                            half_time += '_'
                    else:  # Other drums - sparse
                        if i in [0, 8] and char in ['x', 'X', 'F', 'f']:
                            half_time += char
                        else:
                            half_time += '_'
                else:
                    half_time += '_'
            
            if half_time != '_' * 16:  # Only add non-empty patterns
                half_time_parts.append(f"{drum}: {half_time}")
        
        return "; ".join(half_time_parts)
    
    def _create_improved_double_time_pattern(self, pattern_str: str) -> str:
        """Create improved double-time version with proper musical intensity"""
        if not pattern_str:
            return pattern_str
        
        parts = pattern_str.split(';')
        double_time_parts = []
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip()
            pattern = pattern.strip()
            
            # Normalize to 16 beats first
            pattern = self._normalize_pattern_length(pattern, 16)
            
            # Create double-time feel with increased frequency and intensity
            double_time = ""
            for i in range(16):
                if i < len(pattern):
                    char = pattern[i]
                    # Double-time: more frequent hits, maintain groove
                    if drum == 'bd':  # Bass drum - more frequent
                        if i in [0, 2, 4, 6, 8, 10, 12, 14]:  # 8th note kicks
                            double_time += 'X' if i % 4 == 0 else 'x'
                        else:
                            double_time += '_'
                    elif drum == 'sd':  # Snare - more backbeats
                        if i in [4, 8, 12]:  # More snare hits
                            double_time += 'X' if i == 8 else 'x'
                        elif i in [6, 10, 14]:  # Ghost notes
                            double_time += 'o'
                        else:
                            double_time += '_'
                    elif drum in ['ch', 'hh']:  # Hi-hat - 16th notes
                        if i % 1 == 0:  # All 16th notes
                            double_time += 'x' if char in ['x', 'X', 'o'] else 'o'
                        else:
                            double_time += '_'
                    else:  # Other drums - double original frequency
                        if char in ['x', 'X', 'F', 'f']:
                            double_time += char
                            # Add extra hit on next position if space
                            if i + 1 < 16:
                                continue
                        elif i % 2 == 0 and pattern[i//2] if i//2 < len(pattern) else '_' in ['x', 'X']:
                            double_time += 'x'
                        else:
                            double_time += '_'
                else:
                    double_time += '_'
            
            if double_time != '_' * 16:  # Only add non-empty patterns
                double_time_parts.append(f"{drum}: {double_time}")
        
        return "; ".join(double_time_parts)
    
    def _create_half_time_pattern(self, pattern_str: str) -> str:
        """Create half-time version by placing hits on different beats (still 16 length)"""
        if not pattern_str:
            return pattern_str
        
        parts = pattern_str.split(';')
        half_time_parts = []
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip()
            pattern = pattern.strip()
            
            # Normalize to 16 beats first
            pattern = self._normalize_pattern_length(pattern, 16)
            
            # Create half-time feel by shifting emphasis and reducing density
            half_time = ""
            for i in range(16):
                if i < len(pattern):
                    char = pattern[i]
                    # Half-time: put snare on beat 3 (position 8), kick on 1 and weak positions
                    if i in [0, 8]:  # Strong beats in half-time
                        half_time += char if char != '-' else char
                    elif i in [2, 4, 6, 10, 12, 14]:  # Keep some groove elements
                        half_time += char if char in ['x', 'X', 'o', '_'] else '-'
                    else:
                        half_time += '-'
                else:
                    half_time += '-'
            
            half_time_parts.append(f"{drum}: {half_time}")
        
        return "; ".join(half_time_parts)
    
    def _create_double_time_pattern(self, pattern_str: str) -> str:
        """Create double-time version with more frequent hits (still 16 length)"""
        if not pattern_str:
            return pattern_str
        
        parts = pattern_str.split(';')
        double_time_parts = []
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip()
            pattern = pattern.strip()
            
            # Normalize to 16 beats first
            pattern = self._normalize_pattern_length(pattern, 16)
            
            # Create double-time feel by doubling hit frequency
            double_time = ""
            for i in range(16):
                if i < len(pattern):
                    char = pattern[i]
                    # For double-time, add more frequent hits
                    if char in ['x', 'X', 'o', '_']:
                        double_time += char
                    elif i % 2 == 0 and pattern[i//2] in ['x', 'X', 'o']:
                        # Add extra hits on off-beats for double-time feel
                        double_time += 'x' if char == '-' else char
                    else:
                        double_time += char
                else:
                    double_time += '-'
            
            double_time_parts.append(f"{drum}: {double_time}")
        
        return "; ".join(double_time_parts)
    
    def _normalize_pattern_length(self, pattern: str, target_length: int = 16) -> str:
        """Normalize pattern to target length (default 16 beats)"""
        if len(pattern) == target_length:
            return pattern
        elif len(pattern) > target_length:
            # Truncate to target length
            return pattern[:target_length]
        else:
            # Extend with rests to target length
            return pattern + ('-' * (target_length - len(pattern)))
    
    def _create_quarter_note_pattern(self, pattern_str: str) -> str:
        """Create quarter note version with hits only on strong beats (still 16 length)"""
        if not pattern_str:
            return pattern_str
        
        parts = pattern_str.split(';')
        quarter_parts = []
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip()
            pattern = pattern.strip()
            
            # Normalize to 16 beats first
            pattern = self._normalize_pattern_length(pattern, 16)
            
            # Create quarter note pattern - hits only on beats 1, 2, 3, 4 (positions 0, 4, 8, 12)
            quarter_pattern = ""
            for i in range(16):
                if i in [0, 4, 8, 12]:  # Quarter note positions
                    if i < len(pattern):
                        char = pattern[i]
                        # Keep hits, convert everything else to rest except keep strong accents
                        quarter_pattern += char if char in ['x', 'X', 'o', 'R', '_', '['] else "-"
                    else:
                        quarter_pattern += "-"
                else:
                    quarter_pattern += "-"  # All other positions are rests
            
            quarter_parts.append(f"{drum}: {quarter_pattern}")
        
        return "; ".join(quarter_parts)
    
    def _normalize_pattern_string(self, pattern_str: str) -> str:
        """Normalize all drums in a pattern string to 16 beats"""
        if not pattern_str:
            return pattern_str
        
        parts = pattern_str.split(';')
        normalized_parts = []
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
                
            drum, pattern = part.split(':', 1)
            drum = drum.strip()
            pattern = pattern.strip()
            
            # Normalize each drum pattern to 16 beats
            normalized_pattern = self._normalize_pattern_length(pattern, 16)
            normalized_parts.append(f"{drum}: {normalized_pattern}")
        
        return "; ".join(normalized_parts)
    
    def save_training_data(self, training_data: List[Dict[str, str]], filename: str = "training_data.json"):
        """Save training data to file and generate embeddings
        
        Args:
            training_data: List of training examples
            filename: Output filename
        """
        output_path = self.data_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"💾 Saved {len(training_data)} training examples to {output_path}")
        
        # Generate and save embeddings
        self._generate_and_save_embeddings(training_data)
    
    def load_training_data(self, filename: str = "training_data.json") -> List[Dict[str, str]]:
        """Load training data from file
        
        Args:
            filename: Input filename
            
        Returns:
            List of training examples
        """
        input_path = self.data_dir / filename
        
        if not input_path.exists():
            print(f"❌ Training data file not found: {input_path}")
            return []
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        print(f"📚 Loaded {len(data)} training examples from {input_path}")
        return data
    
    def get_data_statistics(self, training_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get statistics about the training data
        
        Args:
            training_data: List of training examples
            
        Returns:
            Dictionary with data statistics
        """
        if not training_data:
            return {}
        
        df = pd.DataFrame(training_data)
        
        stats = {
            "total_examples": len(training_data),
            "unique_styles": df['style'].nunique() if 'style' in df.columns else 0,
            "style_distribution": df['style'].value_counts().to_dict() if 'style' in df.columns else {},
            "time_signatures": df['time_signature'].value_counts().to_dict() if 'time_signature' in df.columns else {},
            "avg_pattern_length": df['output'].str.len().mean() if 'output' in df.columns else 0
        }
        
        return stats
    
    def load_drum_machine_patterns_260(self, use_converted: bool = True, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Load patterns from DrumMachinePatterns260 - either converted local file or GitHub source
        
        Args:
            use_converted: Use the local converted file with x/- notation instead of GitHub
            force_reload: Whether to re-download even if local file exists
            
        Returns:
            List of drum pattern dictionaries
        """
        if use_converted:
            # Use the new converted file with improved format
            converted_file = self.data_dir / "drum_machine_patterns_260_converted.json"
            
            if converted_file.exists():
                print(f"📂 Loading converted DrumMachinePatterns260 from {converted_file}")
                with open(converted_file, 'r') as f:
                    patterns = json.load(f)
                    
                # Mark patterns as converted format
                for pattern in patterns:
                    pattern['source'] = 'DrumMachinePatterns260_converted'
                    pattern['format'] = 'x_dash_notation'
                    
                print(f"✅ Loaded {len(patterns)} converted patterns")
                return patterns
            else:
                print(f"❌ Converted file not found: {converted_file}")
                print("Falling back to GitHub source...")
        
        # Fallback to original GitHub source
        cache_file = self.data_dir / "drum_machine_patterns_260.json"
        
        if cache_file.exists() and not force_reload:
            print(f"📂 Loading cached DrumMachinePatterns260 from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Direct link to the raw JSON file
        url = "https://raw.githubusercontent.com/stephenhandley/DrumMachinePatterns/master/Sources/DrumMachinePatterns260/Patterns.json"
        
        try:
            print(f"🌐 Downloading DrumMachinePatterns260 from GitHub...")
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Process the data structure - this source has patterns in a different format
            patterns = []
            
            if isinstance(data, dict):
                # If it's a dictionary with pattern categories
                for category, category_patterns in data.items():
                    if isinstance(category_patterns, list):
                        for pattern in category_patterns:
                            pattern['category'] = category
                            pattern['source'] = 'DrumMachinePatterns260'
                            patterns.append(pattern)
                    elif isinstance(category_patterns, dict):
                        for pattern_name, pattern_data in category_patterns.items():
                            if isinstance(pattern_data, dict):
                                pattern_data['name'] = pattern_name
                                pattern_data['category'] = category
                                pattern_data['source'] = 'DrumMachinePatterns260'
                                patterns.append(pattern_data)
            elif isinstance(data, list):
                # If it's directly a list of patterns
                for pattern in data:
                    pattern['source'] = 'DrumMachinePatterns260'
                    patterns.append(pattern)
            else:
                print(f"⚠️  Unexpected data format from {url}")
                return []
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(patterns, f, indent=2)
            
            print(f"✅ Loaded {len(patterns)} patterns from DrumMachinePatterns260")
            return patterns
            
        except Exception as e:
            print(f"❌ Error loading DrumMachinePatterns260: {e}")
            return []
    
    def _generate_and_save_embeddings(self, training_data: List[Dict[str, str]]):
        """Generate embeddings for training data and save to pickle file
        
        Args:
            training_data: List of training examples with 'input' field
        """
        try:
            from sentence_transformers import SentenceTransformer
            print("🧠 Loading embedding model...")
            
            # Use a lightweight, fast model suitable for semantic search
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Extract input texts for embedding
            texts = [example.get('input', '') for example in training_data]
            
            print(f"🧠 Computing embeddings for {len(texts)} training examples...")
            
            # Generate embeddings in batches for efficiency
            embeddings = model.encode(
                texts, 
                batch_size=32, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Save embeddings to pickle file
            embeddings_path = self.data_dir / "training_embeddings.pkl"
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            print(f"💾 Saved embeddings to {embeddings_path}")
            print(f"📊 Embedding shape: {embeddings.shape}")
            print(f"📊 File size: ~{embeddings.nbytes / (1024*1024):.1f} MB")
            
        except ImportError:
            print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")
            print("⚠️  Embeddings not generated. Pattern generation will use keyword matching.")
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            print("⚠️  Pattern generation will use keyword matching fallback.")
    
    def load_embeddings(self, filename: str = "training_embeddings.pkl") -> Optional[np.ndarray]:
        """Load precomputed embeddings from pickle file
        
        Args:
            filename: Embeddings filename
            
        Returns:
            Numpy array of embeddings or None if not found
        """
        embeddings_path = self.data_dir / filename
        
        if not embeddings_path.exists():
            return None
        
        try:
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            
            print(f"📚 Loaded embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return None
