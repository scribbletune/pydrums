"""
MIDI converter for drum patterns
Converts AI-generated patterns to MIDI files
"""

import mido
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


class MidiConverter:
    """Convert drum patterns to MIDI files"""
    
    # General MIDI drum mapping
    DRUM_MIDI_MAP = {
        'bd': 36,    # Bass Drum (Kick)
        'sd': 38,    # Snare Drum
        'ch': 42,    # Closed Hi-Hat
        'oh': 46,    # Open Hi-Hat
        'hh': 44,    # Hi-Hat Pedal
        'cc': 49,    # Crash Cymbal
        'rc': 51,    # Ride Cymbal
        'ht': 50,    # High Tom
        'mt': 47,    # Mid Tom
        'lt': 43,    # Low Tom
        'rim': 37,   # Rim Shot
        'cow': 56,   # Cowbell
        'tamb': 54,  # Tambourine
        'ride': 51,  # Ride Cymbal (alias)
        'crash': 49, # Crash Cymbal (alias)
        'kick': 36,  # Bass Drum (alias)
        'snare': 38, # Snare Drum (alias)
        'hihat': 42, # Closed Hi-Hat (alias)
    }
    
    def __init__(self, output_dir: str = "midi_output"):
        """Initialize MIDI converter
        
        Args:
            output_dir: Directory to save MIDI files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def pattern_to_midi(self, 
                       pattern_data: Dict[str, Any],
                       output_filename: Optional[str] = None,
                       tempo_bpm: int = 120,
                       ticks_per_beat: int = 480,
                       loop_count: int = 4,
                       include_tempo: bool = False,
                       override_speed: Optional[str] = None) -> Path:
        """Convert pattern data to MIDI file
        
        Args:
            pattern_data: Pattern data from PatternGenerator
            output_filename: Optional custom filename
            tempo_bpm: Tempo in beats per minute (only used if include_tempo=True)
            ticks_per_beat: MIDI resolution
            loop_count: Number of times to repeat the pattern
            include_tempo: Whether to include tempo metadata (False = tempo-neutral)
            override_speed: Override the detected speed ('normal', 'half-time', 'double-time', 'quarter')
            
        Returns:
            Path to created MIDI file
        """
        # Determine the speed to use
        speed = override_speed or pattern_data.get('detected_speed', 'normal')
        
        # Calculate timing based on speed
        ticks_per_note = self._calculate_ticks_per_note(ticks_per_beat, speed)
        # Generate filename if not provided
        if not output_filename:
            description = pattern_data.get('description', 'pattern')
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', description)
            output_filename = f"{safe_name}.mid"
        
        output_path = self.output_dir / output_filename
        
        # Extract drum patterns
        drum_patterns = pattern_data.get('drum_patterns', {})
        if not drum_patterns:
            # Try to parse from pattern_line
            pattern_line = pattern_data.get('pattern_line', '')
            drum_patterns = self._parse_pattern_line(pattern_line)
        
        if not drum_patterns:
            raise ValueError("No valid drum patterns found in pattern data")
        
        # Convert to MIDI events with speed-adjusted timing
        events = self._patterns_to_midi_events(
            drum_patterns, ticks_per_note, loop_count
        )
        
        # Create MIDI file
        self._create_midi_file(events, output_path, tempo_bpm, ticks_per_beat, include_tempo)
        
        print(f"💾 MIDI saved: {output_path}")
        return output_path
    
    def _parse_pattern_line(self, pattern_line: str) -> Dict[str, str]:
        """Parse pattern line into drum patterns dictionary"""
        drum_patterns = {}
        
        if not pattern_line:
            return drum_patterns
        
        # Split by semicolon
        parts = pattern_line.split(';') if ';' in pattern_line else [pattern_line]
        
        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue
            
            drum, pattern = part.split(':', 1)
            drum = drum.strip().lower()
            pattern = pattern.strip()
            
            # Clean drum name
            drum_clean = re.sub(r'[^a-z]', '', drum)
            
            if drum_clean and pattern:
                drum_patterns[drum_clean] = pattern
        
        return drum_patterns
    
    def _calculate_ticks_per_note(self, ticks_per_beat: int, speed: Optional[str]) -> int:
        """Calculate ticks per note based on speed setting
        
        Args:
            ticks_per_beat: Base MIDI resolution
            speed: Speed setting ('normal', 'half-time', 'double-time', 'quarter', or None)
            
        Returns:
            Ticks per pattern note position
        """
        speed_multipliers = {
            'half-time': 2,      # Each pattern position = 8th note (slower)
            'double-time': 0.5,  # Each pattern position = 32nd note (faster)  
            'quarter': 4,        # Each pattern position = quarter note (much slower)
            'normal': 1,         # Each pattern position = 16th note (default)
            None: 1              # Default to normal if no speed detected
        }
        
        multiplier = speed_multipliers.get(speed, 1)
        base_ticks_per_16th = ticks_per_beat // 4
        
        return int(base_ticks_per_16th * multiplier)
    
    def _patterns_to_midi_events(self, 
                                drum_patterns: Dict[str, str], 
                                ticks_per_note: int,
                                loop_count: int) -> List[Dict[str, Any]]:
        """Convert drum patterns to MIDI events"""
        events = []
        
        # Find the maximum pattern length to normalize all patterns
        max_length = max(len(pattern) for pattern in drum_patterns.values()) if drum_patterns else 16
        print(f"🎼 Pattern lengths: {[(drum, len(pattern)) for drum, pattern in drum_patterns.items()]}")
        print(f"🎼 Normalizing all patterns to {max_length} beats")
        
        for drum, pattern in drum_patterns.items():
            if drum not in self.DRUM_MIDI_MAP:
                print(f"⚠️  Unknown drum: {drum}, skipping")
                continue
            
            midi_note = self.DRUM_MIDI_MAP[drum]
            
            # Normalize pattern length by padding with rests
            normalized_pattern = self._normalize_pattern_length(pattern, max_length)
            
            # Process each loop
            for loop in range(loop_count):
                loop_offset = loop * max_length * ticks_per_note  # Use normalized length for all patterns
                
                # Process each character in normalized pattern
                for i, char in enumerate(normalized_pattern):
                    tick_time = loop_offset + (i * ticks_per_note)
                    
                    if char == 'x':
                        # Normal hit
                        events.append({
                            'time': tick_time,
                            'note': midi_note,
                            'velocity': 100,
                            'duration': ticks_per_note // 2,
                            'type': 'normal'
                        })
                    elif char == 'R':
                        # Roll - create multiple rapid hits
                        for j in range(4):
                            events.append({
                                'time': tick_time + (j * ticks_per_note // 4),
                                'note': midi_note,
                                'velocity': max(60, 100 - (j * 10)),
                                'duration': ticks_per_note // 8,
                                'type': 'roll'
                            })
                    elif char == '_':
                        # Ghost note - quiet hit
                        events.append({
                            'time': tick_time,
                            'note': midi_note,
                            'velocity': 40,
                            'duration': ticks_per_note // 4,
                            'type': 'ghost'
                        })
                    elif char == '[':
                        # Flam start - grace note before main hit
                        grace_time = max(0, tick_time - (ticks_per_note // 8))  # Ensure non-negative
                        events.append({
                            'time': grace_time,
                            'note': midi_note,
                            'velocity': 60,
                            'duration': ticks_per_note // 8,
                            'type': 'flam_grace'
                        })
                        # Main hit after grace note
                        events.append({
                            'time': tick_time,
                            'note': midi_note,
                            'velocity': 90,
                            'duration': ticks_per_note // 2,
                            'type': 'flam_main'
                        })
                    # ']' and '-' are handled by context or ignored
        
        return sorted(events, key=lambda x: x['time'])
    
    def _normalize_pattern_length(self, pattern: str, target_length: int) -> str:
        """Normalize pattern to target length by padding with rests or truncating
        
        Args:
            pattern: Original pattern string
            target_length: Desired pattern length
            
        Returns:
            Normalized pattern string
        """
        if len(pattern) == target_length:
            return pattern
        elif len(pattern) < target_length:
            # Pad with rests to reach target length
            padding_needed = target_length - len(pattern)
            return pattern + ('-' * padding_needed)
        else:
            # Truncate if pattern is longer than target
            print(f"⚠️  Truncating pattern '{pattern[:10]}...' from {len(pattern)} to {target_length} beats")
            return pattern[:target_length]
    
    def _create_midi_file(self, 
                         events: List[Dict[str, Any]], 
                         output_path: Path,
                         tempo_bpm: int,
                         ticks_per_beat: int,
                         include_tempo: bool = False):
        """Create MIDI file from events"""
        # Create new MIDI file
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Conditionally set tempo
        if include_tempo:
            tempo = mido.bpm2tempo(tempo_bpm)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        # NOTE: If include_tempo=False, MIDI will be tempo-neutral
        # This allows the DAW to control the tempo completely
        
        # Set to drum channel (channel 9, 0-indexed)
        track.append(mido.Message('program_change', channel=9, program=0, time=0))
        
        # Add track name
        track.append(mido.MetaMessage('track_name', name='Drums', time=0))
        
        # Convert events to note_on/note_off pairs
        midi_messages = []
        
        for event in events:
            # Note on at event time
            midi_messages.append({
                'time': event['time'],
                'type': 'note_on',
                'note': event['note'],
                'velocity': event['velocity']
            })
            
            # Note off at event time + duration
            midi_messages.append({
                'time': event['time'] + event['duration'],
                'type': 'note_off',
                'note': event['note'],
                'velocity': 0
            })
        
        # Sort all messages by time
        midi_messages.sort(key=lambda x: x['time'])
        
        # Add messages with proper delta times
        current_time = 0
        
        for msg in midi_messages:
            delta_time = max(0, msg['time'] - current_time)  # Ensure non-negative
            current_time = msg['time']
            
            if msg['type'] == 'note_on':
                track.append(mido.Message(
                    'note_on',
                    channel=9,
                    note=msg['note'],
                    velocity=msg['velocity'],
                    time=delta_time
                ))
            else:  # note_off
                track.append(mido.Message(
                    'note_off',
                    channel=9,
                    note=msg['note'],
                    velocity=msg['velocity'],
                    time=delta_time
                ))
        
        # End of track
        track.append(mido.MetaMessage('end_of_track', time=0))
        
        # Save file
        mid.save(output_path)
    
    def pattern_string_to_midi(self, 
                              pattern_string: str,
                              description: str = "custom_pattern",
                              **kwargs) -> Path:
        """Convert a pattern string directly to MIDI
        
        Args:
            pattern_string: Pattern in format "ch: x-x-; sd: --x-; bd: x---"
            description: Description for filename
            **kwargs: Additional arguments for pattern_to_midi
            
        Returns:
            Path to created MIDI file
        """
        pattern_data = {
            'pattern_line': pattern_string,
            'drum_patterns': self._parse_pattern_line(pattern_string),
            'description': description
        }
        
        return self.pattern_to_midi(pattern_data, **kwargs)
    
    def batch_convert(self, 
                     patterns: List[Dict[str, Any]], 
                     **kwargs) -> List[Path]:
        """Convert multiple patterns to MIDI files
        
        Args:
            patterns: List of pattern data dictionaries
            **kwargs: Additional arguments for pattern_to_midi
            
        Returns:
            List of paths to created MIDI files
        """
        results = []
        
        for i, pattern_data in enumerate(patterns, 1):
            print(f"🎼 Converting {i}/{len(patterns)}: {pattern_data.get('description', 'Unknown')}")
            
            try:
                output_path = self.pattern_to_midi(pattern_data, **kwargs)
                results.append(output_path)
            except Exception as e:
                print(f"❌ Error converting pattern {i}: {e}")
                continue
        
        print(f"✅ Successfully converted {len(results)}/{len(patterns)} patterns")
        return results
    
    def get_midi_info(self, midi_path: Path) -> Dict[str, Any]:
        """Get information about a MIDI file
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Dictionary with MIDI file information
        """
        try:
            mid = mido.MidiFile(midi_path)
            
            info = {
                'filename': midi_path.name,
                'ticks_per_beat': mid.ticks_per_beat,
                'length_seconds': mid.length,
                'num_tracks': len(mid.tracks),
                'num_messages': sum(len(track) for track in mid.tracks),
                'tempo_changes': [],
                'drum_notes': set()
            }
            
            # Analyze tracks
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        bpm = mido.tempo2bpm(msg.tempo)
                        info['tempo_changes'].append(bpm)
                    elif msg.type == 'note_on' and msg.channel == 9:
                        info['drum_notes'].add(msg.note)
            
            info['drum_notes'] = sorted(list(info['drum_notes']))
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def preview_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Create a text preview of the pattern
        
        Args:
            pattern_data: Pattern data dictionary
            
        Returns:
            Formatted text preview
        """
        drum_patterns = pattern_data.get('drum_patterns', {})
        
        if not drum_patterns:
            return "No valid drum patterns found"
        
        preview_lines = [
            f"🎵 Pattern: {pattern_data.get('description', 'Unknown')}",
            f"📊 Valid: {pattern_data.get('is_valid', 'Unknown')}",
            ""
        ]
        
        # Show each drum pattern
        for drum, pattern in drum_patterns.items():
            drum_name = drum.upper()
            midi_note = self.DRUM_MIDI_MAP.get(drum, '?')
            preview_lines.append(f"{drum_name:>6} (#{midi_note:>2}): {pattern}")
        
        return "\\n".join(preview_lines)
