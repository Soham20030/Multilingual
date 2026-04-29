import unicodedata
import re
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)

def segment_sentences(text):
    # Split on Gujarati danda (։), period, question mark, exclamation mark
    sentences = re.split(r'[։.?!]+\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def transliterate_if_needed(sentence):
    # Assuming no transliteration needed for Gujarati script
    return sentence

def deduplicate(sentences):
    return list(OrderedDict.fromkeys(sentences))

def clean_sentence(sentence):
    # Keep Gujarati Unicode range U+0A80 to U+0AFF, common punctuation, digits, spaces
    allowed_pattern = re.compile(r'[\u0A80-\u0AFF\s.,?!\d]+')
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

def process_text_file_in_batches(input_path, batch_size=500000):
    logging.info(f'Start processing input file in batches: {input_path}')
    batch_sentences = []
    batch_number = 1
    lines_read = 0

    def flush_batch(sentences, batch_num):
        output_file = 'gu_train.txt' if batch_num == 1 else 'gu_val.txt'
        process_sentences(sentences, output_file)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines_read += 1
            normalized = normalize_unicode(line)
            segmented = segment_sentences(normalized)
            batch_sentences.extend(segmented)

            if lines_read % 10000 == 0:
                logging.info(f'Read {lines_read} lines so far...')

            if lines_read == batch_size or (batch_number == 2 and lines_read == batch_size * 2):
                logging.info(f'Reached batch {batch_number} size limit ({batch_size} lines).')
                flush_batch(batch_sentences, batch_number)
                batch_sentences = []  # reset for next batch
                batch_number += 1
                if batch_number > 2:
                    break

    # Flush remaining sentences in case fewer lines than expected
    if batch_sentences and batch_number <= 2:
        flush_batch(batch_sentences, batch_number)

    logging.info(f'Total lines read: {lines_read}')
    logging.info('Batch processing complete.')

if __name__ == '__main__':
    process_text_file_in_batches('gu.txt')
