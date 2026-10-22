"""Errors reported by the audio processing stages."""


class AudioConversionError(Exception):
    pass


class NoiseReductionError(Exception):
    pass


class TranscriptionError(Exception):
    pass


class UnsupportedLanguageError(TranscriptionError):
    pass
