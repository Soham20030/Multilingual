import unicodedata
import re
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)

def segment_sentences(text):
    # Split on period, Arabic question mark, exclamation mark, and Urdu danda (if present)
    sentences = re.split(r'[.؟?!]+\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def transliterate_if_needed(sentence):
    # No transliteration needed for Urdu Arabic script
    return sentence

def deduplicate(sentences):
    return list(OrderedDict.fromkeys(sentences))

def clean_sentence(sentence):
    # Arabic Unicode block U+0600–U+06FF, punctuation, digits, spaces allowed
    allowed_pattern = re.compile(r'[\u0600-\u06FF\s.,?!\d]+')
    filtered = ''.join(ch for ch in sentence if allowed_pattern.match(ch))
    filtered = re.sub('\s+', ' ', filtered).strip()
    return filtered

def process_sentences(sentences, output_path):
    logging.info(f'Processing {len(sentences)} sentences for output: {output_path}')
    processed = [clean_sentence(transliterate_if_needed(s)) for s in sentences]
    logging.info('Cleaning and transliteration done.')
    processed = deduplicate(processed)
    logging.info(f'Deduplication done. Unique sentences count: {len(processed)}')
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in processed:
            if sentence:
                f.write(sentence + '\n')
    logging.info(f'Output file {output_path} writing complete.')

def process_text_file_with_sentence_limit(input_path, train_limit=500000, val_limit=500000):
    logging.info(f'Start processing input file: {input_path}')
    train_sentences = []
    val_sentences = []
    lines_read = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines_read += 1
            normalized = normalize_unicode(line)
            segmented = segment_sentences(normalized)
            for sent in segmented:
                if len(train_sentences) < train_limit:
                    train_sentences.append(sent)
                elif len(val_sentences) < val_limit:
                    val_sentences.append(sent)
                else:
                    logging.info(f'Reached sentence limits at line {lines_read}')
                    break
            if len(train_sentences) >= train_limit and len(val_sentences) >= val_limit:
                break
            if lines_read % 10000 == 0:
                logging.info(f'Read {lines_read} lines so far...')

    logging.info(f'Total lines read: {lines_read}')
    process_sentences(train_sentences, 'ur_train.txt')
    process_sentences(val_sentences, 'ur_val.txt')
    logging.info('Processing complete.')

if __name__ == '__main__':
    process_text_file_with_sentence_limit('ur.txt')
