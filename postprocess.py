import re

def post_process_text(chars):
    """
    chars: ['t','h','i','s','i','s','a','t','e','s','t']
    """

    raw_text = "".join(chars)

    # ---- Split words (simple heuristic) ----
    words = raw_text.split()

    processed_words = []

    for i, word in enumerate(words):
        if i == 0:
            processed_words.append(word.capitalize())
        else:
            processed_words.append(word)

    final_text = " ".join(processed_words)

    return final_text
