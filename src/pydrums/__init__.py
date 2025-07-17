"""
PyDrums - AI-powered drum pattern generation and MIDI conversion

A comprehensive toolkit for generating drum patterns using AI and converting them to MIDI.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .pattern_generator import PatternGenerator
from .midi_converter import MidiConverter
from .data_loader import DataLoader

__all__ = ["PatternGenerator", "MidiConverter", "DataLoader"]
