# Whisper Model Parameters Guide 🎯

## Core Generation Parameters

### Temperature (`temperature: float`)
- **Range:** 0.0 to 1.0
- **Default:** 0.6
- **Purpose:** Controls randomness in the output
- **Effects:**
  - Higher (0.7-1.0): More creative, varied, casual outputs
  - Lower (0.2-0.5): More conservative, focused, literal outputs
  - 0.0: Deterministic (always same output)

### Top K (`top_k: int`)
- **Range:** 1 to 100
- **Default:** 30
- **Purpose:** Limits word choice to top K most likely tokens
- **Effects:**
  - Higher (40-100): More diverse vocabulary, casual language
  - Lower (10-20): More conservative word choices, formal language
  - Very low: May sound repetitive or limited

### Top P (`top_p: float`)
- **Range:** 0.0 to 1.0
- **Default:** 0.85
- **Purpose:** Nucleus sampling - dynamically limits token choices
- **Effects:**
  - Higher (0.9-1.0): More varied language, creative
  - Lower (0.5-0.8): More focused, predictable output
  - Very low: May sound too constrained

## Fine-Tuning Parameters

### Repetition Penalty (`repetition_penalty: float`)
- **Range:** 1.0 to 2.0
- **Default:** 1.2
- **Purpose:** Prevents word/phrase repetition
- **Effects:**
  - Higher (1.3-1.5): Strongly avoids repetition
  - Lower (1.0-1.2): Allows natural repetition
  - 1.0: No penalty

### No Repeat NGram Size (`no_repeat_ngram_size: int`)
- **Range:** 0 to 4
- **Default:** 2
- **Purpose:** Prevents repetition of phrases of this length
- **Effects:**
  - Higher (3-4): Prevents longer phrase repetition
  - Lower (1-2): Allows more natural speech patterns
  - 0: Disables feature

### Length Penalty (`length_penalty: float`)
- **Range:** 0.0 to 2.0
- **Default:** 0.8
- **Purpose:** Controls output length tendency
- **Effects:**
  - Higher (>1.0): Favors longer sentences
  - Lower (<1.0): Favors shorter, punchier sentences
  - 1.0: Neutral

### Max New Tokens (`max_new_tokens: int`)
- **Range:** 1 to 512
- **Default:** 256
- **Purpose:** Maximum length of generated text
- **Effects:**
  - Higher: Allows longer outputs
  - Lower: Forces concise responses

## Common Configurations

### Formal Business Setting 
python
generate_kwargs = {
"temperature": 0.3,
"top_k": 20,
"top_p": 0.6,
"repetition_penalty": 1.3,
"no_repeat_ngram_size": 3,
"length_penalty": 1.2,
"max_new_tokens": 256
}
Best for: Professional transcriptions, business meetings, formal presentations

### Casual Conversation
python
generate_kwargs = {
"temperature": 0.6,
"top_k": 30,
"top_p": 0.85,
"repetition_penalty": 1.2,
"no_repeat_ngram_size": 2,
"length_penalty": 0.8,
"max_new_tokens": 256
}
Best for: Daily conversations, informal meetings, podcasts

### Creative/Storytelling
python
generate_kwargs = {
"temperature": 0.8,
"top_k": 50,
"top_p": 0.9,
"repetition_penalty": 1.1,
"no_repeat_ngram_size": 2,
"length_penalty": 1.0,
"max_new_tokens": 384
}
Best for: Creative content, storytelling, entertainment

## Tips for Adjustment
1. Make small changes (0.1 or less) when adjusting parameters
2. Test with the same input multiple times
3. Keep track of which combinations work best for your use case
4. Consider your audience when choosing settings
5. Balance creativity with coherence based on your needs

## Warning Signs & Solutions

### Output Too Random
- **Signs:**
  - Incoherent sentences
  - Off-topic content
  - Nonsensical word combinations
- **Solutions:**
  - Lower temperature (try -0.1)
  - Reduce top_p (try -0.05)
  - Decrease top_k (try -5)

### Too Many Repetitions
- **Signs:**
  - Same phrases appearing frequently
  - Redundant information
  - Circular speaking patterns
- **Solutions:**
  - Increase repetition_penalty (try +0.1)
  - Increase no_repeat_ngram_size
  - Adjust top_k higher

### Unnatural Speech Patterns
- **Signs:**
  - Abrupt sentence endings
  - Overly long sentences
  - Awkward pausing
- **Solutions:**
  - Adjust length_penalty
  - Modify max_new_tokens
  - Fine-tune temperature

### Too Formal
- **Signs:**
  - Robotic language
  - Overly complex vocabulary
  - Stiff sentence structure
- **Solutions:**
  - Increase temperature (+0.1)
  - Increase top_k (+10)
  - Increase top_p (+0.05)

### Too Casual/Incorrect
- **Signs:**
  - Grammar mistakes
  - Slang overuse
  - Poor sentence structure
- **Solutions:**
  - Lower temperature (-0.1)
  - Decrease top_k (-5)
  - Lower top_p (-0.05)

## Best Practices

### For Live Transcription
python
generate_kwargs = {
"temperature": 0.4,
"top_k": 25,
"top_p": 0.75,
"repetition_penalty": 1.2,
"no_repeat_ngram_size": 2,
"length_penalty": 0.9,
"max_new_tokens": 256
}

### For Post-Processing
python
generate_kwargs = {
"temperature": 0.5,
"top_k": 30,
"top_p": 0.8,
"repetition_penalty": 1.3,
"no_repeat_ngram_size": 3,
"length_penalty": 1.0,
"max_new_tokens": 384
}

## Remember
- Always backup your original audio
- Test changes on small samples first
- Keep a log of which settings work best
- Consider your specific use case requirements
- Monitor system performance with higher values

---
*Note: These parameters can be adjusted based on specific needs and may require fine-tuning for optimal results in your particular use case.*