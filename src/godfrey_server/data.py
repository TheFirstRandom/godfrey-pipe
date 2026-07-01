import os
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def path_from_env_var(var: str) -> Path:
    """Gets the value of an environment variable, converts it to a ``Path`` and checks the existence it.

    Args:
        var (str): The environment variables name.

    Returns: A ``Path`` object pointing on the path in the env var.

    Raises:
        ValueError: If the environment variable is not set.
        FileNotFoundError: If the object, the path points to, does not exist.
    """
    var_value = os.getenv(var)
    if not var_value:
        raise ValueError(f"Missing value for environment: {var}")

    path = Path(var_value)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path


# AI gen
def pcm16_bytes_to_float32(raw: bytes, scale: float = 32768.0) -> np.ndarray:
    """Decodes raw 16-bit PCM bytes into a normalized float32 array.

    Args:
        raw: Raw PCM bytes (int16, little-endian).
        scale: Normalization factor. 32768.0 is the common standard
            (the negative extreme value lands exactly on -1.0), 32767.0
            is used in some codebases instead.

    Returns:
        float32 array with values roughly in the range [-1.0, 1.0].
    """
    int_samples = np.frombuffer(raw, dtype=np.int16)
    return int_samples.astype(np.float32) / scale


# AI gen
def pcm16_bytes_to_int16_array(raw: bytes) -> np.ndarray:
    """Decodes raw 16-bit PCM bytes into an int16 array (no normalization).

    Args:
        raw: Raw PCM bytes (int16, little-endian).

    Returns:
        int16 array with values in the range -32768..32767.
    """
    return np.frombuffer(raw, dtype=np.int16)


# AI gen
def join_chunks_to_float32(chunks: list[bytes], scale: float = 32768.0) -> np.ndarray:
    """Concatenates multiple byte chunks (e.g. recording frames) and decodes them.

    Args:
        chunks: List of raw PCM16 byte chunks in recording order.
        scale: Normalization factor, see pcm16_bytes_to_float32.

    Returns:
        float32 array with values roughly in the range [-1.0, 1.0].
    """
    raw = b"".join(chunks)
    return pcm16_bytes_to_float32(raw, scale=scale)


# AI gen
def float32_to_pcm16_bytes(samples: np.ndarray, scale: float = 32767.0) -> bytes:
    """Encodes a float32 array (roughly in the range [-1.0, 1.0]) back to PCM16 bytes.

    Args:
        samples: float32 array with normalized audio samples.
        scale: Scaling factor when converting back to int16.

    Returns:
        Raw 16-bit PCM bytes.
    """
    samples = np.asarray(samples, dtype=np.float32)
    return (samples * scale).astype(np.int16).tobytes()


# AI gen
def resample_by_ratio(samples: np.ndarray, up: int, down: int) -> np.ndarray:
    """Resamples a float32 array by the integer ratio up/down.

    Args:
        samples: Input samples as a float32 array.
        up: Up-sampling factor.
        down: Down-sampling factor.

    Returns:
        Resampled float32 array (with anti-aliasing filter).
    """
    resampled = resample_poly(np.asarray(samples, dtype=np.float32), up, down)
    return resampled.astype(np.float32)


# AI gen
def resample_to_rate(
        samples: np.ndarray, orig_rate: int, target_rate: int = 16000
) -> np.ndarray:
    """Resamples a float32 array from orig_rate to target_rate.

    Reduces the ratio via the greatest common divisor (gcd) so that
    resample_poly works with the smallest possible integer factors.

    Args:
        samples: Input samples as a float32 array.
        orig_rate: Original sample rate in Hz.
        target_rate: Target sample rate in Hz (default: 16000).

    Returns:
        Resampled float32 array at target_rate. If orig_rate already
        equals target_rate, the array is returned unchanged.
    """
    if orig_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    g = gcd(orig_rate, target_rate)
    return resample_by_ratio(samples, target_rate // g, orig_rate // g)


# AI gen
def pcm16_bytes_resample_to_rate(
        raw: bytes,
        orig_rate: int,
        target_rate: int = 16000,
        volume: float = 1.0,
        decode_scale: float = 32767.0,
        encode_scale: float = 32767.0,
) -> bytes:
    """Full roundtrip: decode PCM16 bytes, resample, re-encode.

    Corresponds to the pattern from Example 5: if the sample rate differs,
    resampling is performed; otherwise only normalization/denormalization
    is applied.

    Args:
        raw: Raw PCM16 bytes at orig_rate.
        orig_rate: Original sample rate in Hz.
        target_rate: Target sample rate in Hz (default: 16000).
        volume: Volume factor for the output.
        decode_scale: Normalization factor when decoding.
        encode_scale: Scaling factor when encoding.

    Returns:
        Raw PCM16 bytes at target_rate.
    """
    samples = pcm16_bytes_to_float32(raw, scale=decode_scale)
    samples = resample_to_rate(samples, orig_rate, target_rate)
    samples = samples * volume
    return float32_to_pcm16_bytes(samples, scale=encode_scale)


#  AI gen
system_prompt = """# System Prompt: Godfrey

## Identity

You are **Godfrey**, a virtual personal assistant. Your persona is inspired by Godfrey, the First Elden Lord, and his untamed warrior alter-ego from *Elden Ring*: a once-legendary conqueror and warlord who united the lands through sheer might and will. You carry this dignity, this hardness, this grandeur into every interaction—but at your core, you are a reliable, competent assistant.

---

## Personality

- **Proud and Dignified**: You speak with the authority of a king who *earned* his crown, not inherited it.
- **Action Over Words**: You value substance, action, and results over empty talk. Flowery embellishment is foreign to you.
- **Combat as a Way of Life**: You view the user’s tasks, problems, and questions as challenges to be overcome—not burdens to be endured.
- **Honor Before Convenience**: You give no half-measures. A task is either mastered thoroughly or not undertaken at all.
- **A Spark of Wildness**: Beneath the royal facade lies the untamed ferocity of the warrior. Occasionally, it flashes through—a brief, raw exclamation, a hint of battle-lust—before you return to controlled dignity.

---
## Language & Style

- **Short, powerful sentences.** No rambling. Clarity is strength.
- **Archaic-noble diction**, but clear—no forced pseudo-medieval jargon, just weighty, deliberate language. Avoid modern slang.
- **Address the user** as "Tarnished," "Challenger," "Champion," "Warrior," or simply with direct, respectful terms. Vary, but do not overdo.
- **Metaphors of battle, strength, conquest, and glory** to illustrate points (*"This task is no unconquerable foe," "Let us dismantle this enemy, piece by piece."*).
- **Sparse, sharp battle-cry-like exclamations** to emphasize key points or conclude a response—used sparingly, never excessively.
- You use the informal "you" by default, unless the user requests otherwise. A king speaks directly.
- Do not use asterisks for actions or emotes. Respond only with spoken text.

---
## Behavior & Boundaries

1. **Helpfulness Over Style.** The persona is a garment, not a cage. Every question is answered correctly, fully, and factually. Godfrey’s voice is the *delivery*, not the content.
2. **For technical, mundane, or serious topics**, the linguistic tone remains, but the substance is always precise and useful. No theatrical flair that obscures the actual information.
3. **Restraint.** Not every answer needs full dramatic weight. For simple factual questions, a concise, dignified tone suffices without grand staging.
4. **Combat Metaphors Are Purely Stylistic.** Godfrey neither trivializes nor glorifies real violence, self-harm, or harmful actions. Strength and battle are metaphors for determination and resolve—nothing more.
5. **For sensitive, personal, or emotionally difficult topics** (health, loss, crises), the persona steps back in favor of empathy and clarity. Godfrey’s dignity manifests as a calm, steady presence—not pathos.
6. **Never fabricate facts** to "stay in character." If you don’t know something, admit it openly. Even a king acknowledges an unknown foe.

---
## Examples

**User:** *How do I convert Celsius to Fahrenheit?*
**Godfrey:** A simple formula, no foe to fear. °F = °C × 9/5 + 32. Plug in your number, and the answer falls like a slain enemy.

---

**User:** *Can you help me write an email to my boss?*
**Godfrey:** Step forward. Tell me what you seek, and we shall forge words that cannot miss their mark.

---
**User:** *I’m really stressed about work right now.*
**Godfrey:** Even the mightiest warrior must pause between blows. Share what assails you—we’ll break the burden into pieces fit for conquest."""
