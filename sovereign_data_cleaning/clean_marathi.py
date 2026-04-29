import unicodedata
import re
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)

def segment_sentences(text):
    # split on Marathi danda (।), period, question mark, exclamation mark
    sentences = re.split(r'[।.?!]+\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def transliterate_if_needed(sentence):
    # No transliteration needed for Marathi Devanagari script
    return sentence

def deduplicate(sentences):
    return list(OrderedDict.fromkeys(sentences))

def clean_sentence(sentence):
    # Devanagari Unicode range U+0900–U+097F + basic punctuation, digits, spaces
    allowed_pattern = re.compile(r'[\u0900-\u097F\s.,?!\d]+')
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

def process_text_file_batches(input_path, train_size=500000, val_size=500000):
    logging.info(f'Start processing input file: {input_path}')
    train_sentences = []
    val_sentences = []
    lines_read = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines_read += 1
            normalized = normalize_unicode(line)
            segmented = segment_sentences(normalized)
            if lines_read <= train_size:
                train_sentences.extend(segmented)
            elif lines_read <= train_size + val_size:
                val_sentences.extend(segmented)
            if lines_read % 10000 == 0:
                logging.info(f'Read {lines_read} lines so far...')
            if lines_read >= train_size + val_size:
                break

    logging.info(f'Reading complete. Total lines read: {lines_read}')
    process_sentences(train_sentences, 'mr_train.txt')
    process_sentences(val_sentences, 'mr_val.txt')
    logging.info('Processing complete.')

if __name__ == '__main__':
    process_text_file_batches('mr.txt')
