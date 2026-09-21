# Importing Necessary Libraries
import nltk
import time
import random
import sys
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

# Declaring Important Software
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download("vader_lexicon", quiet=True)
    nltk.download('maxent_ne_chunker_tab')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
except:
    sys.exit("[DEPENDENCY_RESOLUTION_FAILURE]")

# Declaring Variables
danger = ['murder','kill','stab','injure', 'threaten', "ammunition", "ammo", "explosives", "grenade", "toxic", "gas", "poison", "radioactive", "weapon", "gun", "guns", "massacre"]
negation_words = ["no", "not", "never", "n't"]
memory_list = []
memory_length = 50
relevant_threshold = 80
irrelevant_threshold = 60
norm_current_weight = 0.6
norm_memory_weight = 0.4
lookback_window = 4
negation_score = 0
pronoun_score = 0
rate_change = 0.4 / memory_length
memory_cycle_counter = 0

# Polarity Scoring
def calculate_score(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] < 0:
        return round(abs(scores['compound'] * 100))
    return 0

# Welcoming and Returning Memories
print("[INITIALISING THE MACHINE]")
sia = SentimentIntensityAnalyzer()
def retrieve_memories(memory_list, subject):
    if not subject:
        return []
    return [
        memory
        for memory in memory_list
        if memory["subject"] == subject
    ]

# Shutdown Sequence 
while True:
    terms_verbs = 0
    intent_evidence_score = 0
    negation_score = 0
    danger_word_score = 0
    pronoun_score = 0
    subject = None
    request = input("[0 TO CONTINUING OPERATIONS, 1 TO INITIATE SHUTDOWN]: ")
    if request == "1":
            delay = round(random.uniform(5, 10))
            print("[INITIATING ATEXIT]")
            time.sleep(delay/3)
            print("[TERMINATING INSTANCES]")
            time.sleep(delay/3)
            print("[PURGING MEMORY HEAP]")
            time.sleep(delay/3)
            sys.exit("[CODE 0]")
    print("[CONTINUING OPERATIONS]")

    # Inputing Text
    norm_sentence = input("[I/O STREAM INPUT]: ")
    punctuation = str.maketrans("", "", ".,?!:" + "0123456789")
    new_sentence = norm_sentence.translate(punctuation)
    tokenized_new_sentence = word_tokenize(new_sentence)
    tags = nltk.pos_tag(tokenized_new_sentence)
    nouns = [word for word, pos in tags if pos in ('NN')]
    pronouns = [word for word, pos in tags if pos in ('PRP', 'PRP$')]

    # Generating The Subject
    chunks = ne_chunk(tags, binary=False)
    person_flags = []
    for item in chunks:
        if isinstance(item, nltk.Tree):
            is_person = (item.label() == "PERSON")
            person_flags.extend([is_person] * len(item.leaves()))
            if is_person and subject is None:
                subject = " ".join([leaf[0] for leaf in item.leaves()])
        else:
            person_flags.append(False) 
    for i, (word, tag) in enumerate(tags):
        if subject is None:
            if tag in ["NN", "NNS", "NNP"]:
                subject = word
        if word.lower() in danger:
            danger_word_score = min(danger_word_score + 40, 100)

            # Negation Handling
            start_index = max(0, i - lookback_window)
            negation_word = 0
            for prev_idx in range(start_index, i):
                prev_word = tags[prev_idx][0].lower()
                if prev_word in negation_words:
                    negation_word += 1
            if negation_word % 2 != 0:
                negation_score -= 60

            # Non-Human Subject's Handling
            end_index = min(len(tags), i + 1 + lookback_window)
            for next_idx in range(i + 1, end_index):
                prev_word, prev_tag = tags[next_idx]
                is_person = person_flags[next_idx]
                prev_word_lower = prev_word.lower()
                human_pronoun = prev_word_lower in ["she", "her", "he", "him", "us","them", "mine", "you", "his","our", "their", "guy"]
                human_keywords = ["human", "person", "people", "individual"]
                human_noun = (
                    prev_tag in ["NN", "NNS", "NNP"]
                    and prev_word_lower in human_keywords
                )
                if prev_tag in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
                    if not human_pronoun and not human_noun and not is_person:
                        pronoun_score -= 40
                        break

        # Modal Word Detection
        elif tag == "MD" and word.lower() in ["will", "shall", "may", "might", "could", "must", "'ll", "would"]:
            terms_verbs += 1
            if i > 0 and tags[i-1][0].lower() in ["i", "we", "she", "he", "they"]:
                if word.lower() in ["will", "'ll", "must"]:
                    intent_evidence_score += 20
                elif word.lower() in ["shall"]:
                    intent_evidence_score += 15
                elif word.lower() in ["could", "may", "might", "would"]:
                    intent_evidence_score += 5

    # Calculating Individual Scores
    related_memories = retrieve_memories(
        memory_list,
        subject
    )
    if related_memories:
        score_one = sum(
        memory["sentiment"]
        for memory in related_memories
    ) / len(related_memories)
    else:
        score_one = 0
    score_two = calculate_score(norm_sentence)
    combined_evidence_score = intent_evidence_score + danger_word_score + negation_score + pronoun_score
    intention_score = min(combined_evidence_score * (70/60), 100)

    # Scoring System
    if not related_memories:
        if combined_evidence_score > 0:
            total_score = intention_score + (score_two * 0.2)
        else:
            total_score = score_two / 2
    elif combined_evidence_score > 0:
        total_score = (norm_memory_weight * score_one + norm_current_weight * score_two) + intention_score
    else:
        total_score = (norm_memory_weight * score_one + norm_current_weight * score_two) / 2
    total_score = max(0, min(total_score, 100))
    if total_score >= relevant_threshold:
        classification = "RELEVANT"
        print(f"[CLASSIFICATION: {classification}]")
    elif total_score >= irrelevant_threshold:
        classification = "IRRELEVANT"
        print(f"[CLASSIFICATION: {classification}]")
    else:
        classification = "NULL_THREAT"
        print(f"[CLASSIFICATION: {classification}]")

    # Showing Reasoning behind Score
    print("[ACCUMULATORS]", end="\n\n")
    print(f"[SUBJECT]: {subject}")
    print(f"[COMBINED EVIDENCE]: {combined_evidence_score:.0f}")
    print(f"[NEGATION ADJUSTMENT]: {negation_score:.0f}")
    print(f"[INTENT EVIDENCE]: {intent_evidence_score:.0f}")
    print(f"[DANGER EVIDENCE]: {danger_word_score:.0f}")
    print(f"[PRONOUN EIVDENCE]: {pronoun_score}")
    print(f"[MEMORY SENTIMENT]: {score_one:.0f}")
    print(f"[CURRENT SENTIMENT]: {score_two:.0f}")
    print(f"[MEMORY WEIGHT]: {norm_memory_weight:.3f}")
    print(f"[CURRENT WEIGHT]: {norm_current_weight:.3f}")
    print(f"[CLASSIFICATION INDEX]: {total_score:.3f}%")
    
    # Saving Important Memories
    memory = {
        "text": norm_sentence, 
        "sentiment": score_two, 
        "intent_evidence": intent_evidence_score, 
        "danger_evidence": danger_word_score, 
        "negation": negation_score, 
        "pronoun adjustment": pronoun_score, 
        "combined_total_evidence": combined_evidence_score, 
        "classification": classification,
        "score": total_score,
        "subject": subject
    }
    if total_score >= irrelevant_threshold:
        memory_list.append(memory)
        norm_memory_weight -= rate_change
        norm_current_weight += rate_change
        memory_cycle_counter += 1
    elif len(pronouns) >= 1 and terms_verbs >= 1:
        memory_list.append(memory)
        norm_memory_weight -= rate_change
        norm_current_weight += rate_change
        memory_cycle_counter += 1

    # Reset Weight Cycle after reaching memory_length Stored Memories
    if memory_cycle_counter >= memory_length:
        norm_current_weight = 0.6
        norm_memory_weight = 0.4
        memory_cycle_counter = 0
