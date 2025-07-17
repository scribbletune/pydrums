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
        """Convert drum data to pattern notation string"""
        if not drums:
            return ""
        
        # Map drum names to abbreviations
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
            'LowTom': 'lt'
        }
        
        pattern_parts = []
        
        for drum_name, hits in drums.items():
            if drum_name in drum_mapping and hits:
                notation = self._hits_to_notation(hits)
                if notation:
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
        """Generate multiple training examples from one pattern"""
        examples = []
        
        # Base prompts
        prompts = [
            f"Create a {style} drum pattern",
            f"Generate a {style} beat",
            f"Make a {style} drum loop",
            f"Create a {time_sig} {style} pattern",
        ]
        
        # Add time signature specific prompts
        if time_sig == "4/4":
            prompts.append(f"Generate a standard {style} beat")
        elif time_sig == "12/8":
            prompts.append(f"Create a {style} shuffle pattern")
        
        # Create training pairs
        for prompt in prompts:
            examples.append({
                "input": prompt,
                "output": pattern_str,
                "style": style,
                "time_signature": time_sig,
                "source_pattern": name
            })
        
        return examples
    
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
