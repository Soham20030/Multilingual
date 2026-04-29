import unicodedata
import re
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)


def segment_sentences(text):
    sentences = re.split(r'[।.?!]+\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def transliterate_if_needed(sentence):
    return sentence  # placeholder – no transliteration needed for Bodo right now


def deduplicate(sentences):
    return list(OrderedDict.fromkeys(sentences))


def clean_sentence(sentence):
    # ✅ FIX: use Devanagari Unicode range (\u0900-\u097F)
    allowed_pattern = re.compile(r'[\u0900-\u097F\s.,?!\d]+')
    filtered = ''.join(ch for ch in sentence if allowed_pattern.match(ch))
    filtered = re.sub(r'\s+', ' ', filtered).strip()
    return filtered


def process_sentences(sentences, output_path):
    logging.info(f'Processing {len(sentences)} sentences for output: {output_path}')
    processed = [clean_sentence(transliterate_if_needed(s)) for s in sentences]
    logging.info('Cleaning and transliteration done.')
    processed = deduplicate(processed)
    logging.info(f'Deduplication done. Unique sentences count: {len(processed)}')

    logging.info(f'Writing output to: {output_path}')
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in processed:
            if sentence:
                f.write(sentence + '\n')
    logging.info(f'Output file {output_path} writing complete.')


def process_text_file_split_in_half(input_path):
    logging.info(f'Start processing input file: {input_path}')

    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    logging.info(f'Total lines in file: {total_lines}')

    half_lines = total_lines // 2
    batch_sentences = []
    batch_number = 1
    lines_read = 0

    def flush_batch(sentences, batch_num):
        output_file = 'bd_train.txt' if batch_num == 1 else 'bd_val.txt'
        process_sentences(sentences, output_file)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines_read += 1
            normalized = normalize_unicode(line)
            segmented = segment_sentences(normalized)
            batch_sentences.extend(segmented)

            if lines_read % 10000 == 0:
                logging.info(f'Read {lines_read} lines so far...')

            if (batch_number == 1 and lines_read == half_lines) or (batch_number == 2 and lines_read == total_lines):
                logging.info(f'Reached half point or end for batch {batch_number}.')
                flush_batch(batch_sentences, batch_number)
                batch_sentences = []
                batch_number += 1
                if batch_number > 2:
                    break

    if batch_sentences and batch_number <= 2:
        flush_batch(batch_sentences, batch_number)

    logging.info(f'Total lines read: {lines_read}')
    logging.info('Half split batch processing complete.')


if __name__ == '__main__':
    process_text_file_split_in_half('bd.txt')
