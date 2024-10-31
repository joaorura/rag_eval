import nltk
from nltk.translate.bleu_score import sentence_bleu

# Define reference and candidate sentences
reference = [['this', 'is', 'a', 'test'], ['this', 'is', 'test']]
candidate = ['this', 'is', 'a', 'test']

# Calculate BLEU scores with different n-gram weights
bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))  # Unigram
bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))  # Bigram
bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))  # Trigram
bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))  # 4-gram

print(f'BLEU-1: {bleu1}')
print(f'BLEU-2: {bleu2}')
print(f'BLEU-3: {bleu3}')
print(f'BLEU-4: {bleu4}')