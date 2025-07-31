"""
Data loader for drum pattern datasets
Handles loading and processing of training data from various sources
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


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
    
    def _convert_drums_to_notation(self, drums: Dict[str, List]) -> str:
        """Convert drum data to pattern notation string with enhanced musical representation"""
        if not drums:
            return ""
        
        # Enhanced drum mapping with more instruments
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
    
    def _hits_to_notation(self, hits: List[str]) -> str:
        """Convert hit list to notation string"""
        notation_map = {
            'Note': 'x',
            'Rest': '-'
        }
        
        return "".join(notation_map.get(hit, '-') for hit in hits)
    
    def _generate_training_examples(self, pattern_str: str, style: str, 
                                  time_sig: str, name: str) -> List[Dict[str, str]]:
        """Generate diverse training examples from one pattern with musical variation"""
        examples = []
        
        # Create more varied and musically descriptive prompts
        style_descriptors = {
            'rock': ['heavy', 'driving', 'steady', 'powerful', 'straight'],
            'funk': ['syncopated', 'groovy', 'tight', 'pocket', 'with ghost notes'],
            'jazz': ['swinging', 'brushed', 'subtle', 'loose', 'walking'],
            'disco': ['four on the floor', 'danceable', 'steady', 'pumping'],
            'reggae': ['one drop', 'laid back', 'off-beat', 'skank'],
            'latin': ['syncopated', 'complex', 'polyrhythmic', 'spicy'],
            'afro': ['polyrhythmic', 'complex', 'traditional', 'layered'],
            'blues': ['shuffled', 'swinging', 'laid back', 'soulful'],
            'ballad': ['gentle', 'soft', 'slow', 'emotional'],
            'pop': ['catchy', 'simple', 'accessible', 'radio-friendly']
        }
        
        descriptors = style_descriptors.get(style, ['basic', 'simple', 'standard'])
        
        # Generate 2-3 examples per pattern (reduced from 5+ identical ones)
        base_prompts = [
            f"Create a {style} drum pattern",
            f"Generate a {descriptors[0]} {style} beat"
        ]
        
        # Add time signature and descriptor-specific prompts
        if time_sig == "4/4":
            base_prompts.append(f"Make a {descriptors[1] if len(descriptors) > 1 else 'standard'} {style} groove")
        elif time_sig == "12/8":
            base_prompts.append(f"Create a {style} shuffle with {descriptors[0]} feel")
        elif time_sig == "6/8":
            base_prompts.append(f"Generate a {style} pattern in 6/8 time")
        
        # Only create 2-3 base examples instead of 5+
        for i, prompt in enumerate(base_prompts[:3]):
            examples.append({
                "input": prompt,
                "output": pattern_str,
                "style": style,
                "time_signature": time_sig,
                "source_pattern": name,
                "speed": "normal",
                "pattern_length": len(pattern_str.split(';')[0].split(':')[1].strip()) if ':' in pattern_str else 16,
                "variation": f"base_{i+1}"
            })
        
        # Generate speed variation examples (but fewer of them)
        speed_variations = self._generate_speed_variations(pattern_str, style, time_sig, name)
        examples.extend(speed_variations[:6])  # Limit speed variations
        
        return examples
    
    def _generate_speed_variations(self, pattern_str: str, style: str, 
                                 time_sig: str, name: str) -> List[Dict[str, str]]:
        """Generate speed variation training examples with better musical accuracy"""
        variations = []
        
        # Half-time variations (double length, half speed) - only 2 examples
        half_time_pattern = self._create_half_time_pattern(pattern_str)
        if half_time_pattern and half_time_pattern != pattern_str:
            half_time_prompts = [
                f"Create a half-time {style} groove",
                f"Generate a laid-back {style} pattern"
            ]
            
            for prompt in half_time_prompts:
                variations.append({
                    "input": prompt,
                    "output": half_time_pattern,
                    "style": style,
                    "time_signature": time_sig,
                    "source_pattern": f"{name}_halftime",
                    "speed": "half_time",
                    "pattern_length": len(half_time_pattern.split(';')[0].split(':')[1].strip()) if ':' in half_time_pattern else 32,
                    "variation": "half_time"
                })
        
        # Double-time variations (half length, double speed) - only 2 examples
        double_time_pattern = self._create_double_time_pattern(pattern_str)
        if double_time_pattern and double_time_pattern != pattern_str:
            double_time_prompts = [
                f"Create a double-time {style} beat",
                f"Generate a fast {style} rhythm"
            ]
            
            for prompt in double_time_prompts:
                variations.append({
                    "input": prompt,
                    "output": double_time_pattern,
                    "style": style,
                    "time_signature": time_sig,
                    "source_pattern": f"{name}_doubletime",
                    "speed": "double_time", 
                    "pattern_length": len(double_time_pattern.split(';')[0].split(':')[1].strip()) if ':' in double_time_pattern else 8,
                    "variation": "double_time"
                })
        
        # Quarter note variations (simpler, emphasize strong beats) - only 2 examples
        quarter_pattern = self._create_quarter_note_pattern(pattern_str)
        if quarter_pattern and quarter_pattern != pattern_str:
            quarter_prompts = [
                f"Create a simple {style} pattern with quarter notes",
                f"Generate a basic {style} beat on strong beats"
            ]
            
            for prompt in quarter_prompts:
                variations.append({
                    "input": prompt,
                    "output": quarter_pattern,
                    "style": style,
                    "time_signature": time_sig,
                    "source_pattern": f"{name}_quarter",
                    "speed": "quarter_notes",
                    "pattern_length": len(quarter_pattern.split(';')[0].split(':')[1].strip()) if ':' in quarter_pattern else 4,
                    "variation": "quarter_notes"
                })
        
        return variations
    
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
        """Save training data to file
        
        Args:
            training_data: List of training examples
            filename: Output filename
        """
        output_path = self.data_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"💾 Saved {len(training_data)} training examples to {output_path}")
    
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
    
    def load_drum_machine_patterns_260(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Load patterns from DrumMachinePatterns260 GitHub source
        
        Args:
            force_reload: Whether to re-download even if local file exists
            
        Returns:
            List of drum pattern dictionaries
        """
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
