import os
import re
from openai import AzureOpenAI
from typing import List, Dict, Optional, Tuple
import argparse
import logging

class TextFileTranslator:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str = "2025-03-01-preview",
                 deployment_name: str = "gpt-4", max_line_length: int = 80):
        """
        Initialize the translator with Azure OpenAI credentials
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.max_line_length = max_line_length

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Patterns for text that should NOT be translated
        self.non_translatable_patterns = [
            r'^[\d\s\-\(\)\/\.\,\:\;\+\=\*\#\$\%\^\&\|\~\`\[\]\{\}\_]+$',  # Only numbers/symbols
            r'^[A-Z]{1,3}$',  # Short abbreviations (Y, N, M, F, etc.)
            r'^NML\d+$',  # Form numbers like NML5
            r'^DFA-[A-Z]+-\d+$',  # Form codes like DFA-LTC-5
            r'^\{[^}]*\}$',  # Checkbox patterns like { }
            r'^[\s]*$',  # Only whitespace
            r'^\([^)]*\)$',  # Content in parentheses like phone number format
            r'^[\d\-\(\)\s]+$',  # Phone numbers and dates
            r'^\w{1,2}$',  # Single letters or very short codes
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.non_translatable_patterns]

        # Common Y/N replacements
        self.yn_replacements = {
            'Y': 'S',
            'N': 'N',
            'YES': 'SÍ',
            'NO': 'NO',
            'M': 'M',
            'F': 'F'
        }



    def is_form_header_or_code(self, text: str) -> bool:
        """Check if text is a form header or code that should be preserved"""
        text_upper = text.upper().strip()

        # Form codes and numbers
        if re.match(r'^(NML\d+|DFA-[A-Z]+-\d+)$', text_upper):
            return True

        # Single letters used as codes (but we'll handle Y/N specially)
        if re.match(r'^[A-Z]$', text_upper) and text_upper not in ['Y', 'N']:
            return True

        # Checkbox patterns
        if re.match(r'^\{[^}]*\}$', text.strip()):
            return True

        return False

    def should_translate_text(self, text: str, context_line: str = "") -> bool:
        """
        Determine if text should be translated based on content and context
        """
        if not text or len(text.strip()) == 0:
            return False

        text_stripped = text.strip()

        # Check if it's a form code or header
        if self.is_form_header_or_code(text_stripped):
            return False

        # Check against non-translatable patterns
        for pattern in self.compiled_patterns:
            if pattern.match(text_stripped):
                return False

        # Must contain at least one alphabetic character
        if not re.search(r'[A-Za-z]', text_stripped):
            return False

        # Don't translate very short sequences unless they're common words
        if len(text_stripped) <= 2:
            return False

        # Skip segments that are simply phone numbers or parentheses
        if re.match(r'^[\d\-\(\)\s]+$', text_stripped):
            return False

        return True

    def extract_translatable_content(self, line: str) -> List[Dict]:
        """
        Extract translatable content using a more intelligent approach
        """
        segments = []

        # Skip lines that are mostly formatting or codes
        if re.match(r'^[\s\-\=\*\#]+$', line):
            return segments

        # Find meaningful text segments
        # Look for sequences of words that form coherent phrases
        word_sequences = re.finditer(r'[A-Za-z][A-Za-z\s\-\.\,\(\)\/\:\'\?\!]*[A-Za-z]', line)

        for match in word_sequences:
            text = match.group().strip()

            # Skip if too short or should not be translated
            if len(text) < 3 or not self.should_translate_text(text, line):
                continue

            # Check if this is a meaningful phrase worth translating
            word_count = len(text.split())
            if word_count >= 2 or len(text) >= 4:  # Multi-word phrases or longer single words
                segments.append({
                    'text': text,
                    'start': match.start(),
                    'end': match.end(),
                    'original': text
                })

        return segments

    def handle_yn_replacements(self, text: str) -> str:
        """Handle Y/N replacements in text"""
        result = text
        for eng, esp in self.yn_replacements.items():
            # Replace standalone Y/N with proper spacing
            result = re.sub(r'\b' + eng + r'\b', esp, result)
        return result

    def translate_text_with_context(self, text: str, context: str = "", retries: int = 1) -> str:
        """Translate text with context awareness and retry if the text is not translated"""
        try:
            clean_text = text.strip()

            # Handle Y/N replacements first
            if clean_text.upper() in self.yn_replacements:
                return self.yn_replacements[clean_text.upper()]


            # Build context-aware prompt
            context_info = ""
            if context:
                context_info = f"\n\nContext (the line this text appears in): {context}"

            base_prompt = f"""You are an expert translator specializing in translating official government and medical forms from English to Spanish.

CRITICAL TRANSLATION RULES:
1. Translate the given English text to formal, official Spanish
2. Maintain exact capitalization style (ALL CAPS → ALL CAPS, Title Case → Title Case)
3. Use standard government/medical terminology appropriate for official documents
4. Keep similar length when possible to preserve form formatting
5. For form fields and labels, use conventional Spanish translations used in official documents
6. Be consistent with translations throughout the document
7. Consider the context to provide the most appropriate translation
8. For Y/N indicators: Y = S (Sí), N = N (No)
9. For YES/NO: YES = SÍ, NO = NO
10. Translate references to ethnicities or nationalities into Spanish

FORMATTING RULES:
- If input is ALL CAPS, output must be ALL CAPS
- If input is Title Case, output must be Title Case
- If input is lowercase, output must be lowercase
- Preserve any punctuation and spacing patterns

Return ONLY the Spanish translation with no explanations or additional text.{context_info}"""

            system_prompt = base_prompt

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate to Spanish: {clean_text}"}
                ],
                temperature=0.1,
                max_tokens=200
            )

            translation = response.choices[0].message.content.strip()

            # Clean up translation
            translation = self._clean_translation_output(translation)

            # If translation looks unchanged, retry with stronger instruction
            if translation.strip().upper() == clean_text.upper() and retries > 0:
                retry_prompt = base_prompt + "\n\nIf any part of the text is in English, ensure it is translated to Spanish." + context_info
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": retry_prompt},
                        {"role": "user", "content": f"Translate completely to Spanish: {clean_text}"}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                translation = response.choices[0].message.content.strip()
                translation = self._clean_translation_output(translation)

            # Ensure capitalization is preserved
            translation = self._preserve_capitalization(clean_text, translation)

            return translation

        except Exception as e:
            self.logger.error(f"Translation error for '{text}': {e}")
            return text

    def _clean_translation_output(self, translation: str) -> str:
        """Clean up translation output"""
        # Remove quotes if they wrap the entire translation
        if ((translation.startswith('"') and translation.endswith('"')) or
            (translation.startswith("'") and translation.endswith("'"))):
            translation = translation[1:-1]

        # Remove common prefixes
        prefixes = ["Spanish: ", "Translation: ", "Translated: ", "En español: ", "Spanish translation: "]
        for prefix in prefixes:
            if translation.lower().startswith(prefix.lower()):
                translation = translation[len(prefix):]
                break

        return translation.strip()

    def _preserve_capitalization(self, original: str, translation: str) -> str:
        """Preserve the capitalization style of the original text"""
        if original.isupper():
            return translation.upper()
        elif original.islower():
            return translation.lower()
        elif original.istitle():
            return translation.title()
        else:
            return translation

    def _detect_yn_positions(self, line: str) -> List[int]:
        """Return index positions of standalone Y/N indicators"""
        return [m.start() for m in re.finditer(r'(?<!\w)[YN](?!\w)', line)]

    def _align_yes_no_spacing(self, original_line: str, translated_line: str, positions: List[int]) -> str:
        """Align S/N indicators to the same positions as the original Y/N"""
        if not positions:
            return translated_line

        translated_positions = [m.start() for m in re.finditer(r'(?<!\w)[SN](?!\w)', translated_line)]
        if not translated_positions:
            return translated_line

        chars = list(translated_line)
        for orig_pos, trans_pos in zip(positions, translated_positions):
            diff = orig_pos - trans_pos
            if diff > 0:
                chars.insert(trans_pos, ' ' * diff)
            elif diff < 0:
                start = max(0, trans_pos + diff)
                del chars[start:trans_pos]
            translated_line = ''.join(chars)
            translated_positions = [m.start() for m in re.finditer(r'(?<!\w)[SN](?!\w)', translated_line)]
            if len(translated_positions) < len(positions):
                break

        return translated_line

    def merge_paragraph_lines(self, lines: List[str]) -> List[str]:
        """Merge consecutive lines that form a single paragraph"""
        merged = []
        buffer = ""
        prefix = ""

        for line in lines:
            stripped = line.rstrip("\n\r")

            if not stripped.strip():
                if buffer:
                    merged.append(prefix + buffer)
                    buffer = ""
                    prefix = ""
                merged.append(stripped)
                continue

            current_prefix = re.match(r'^\s*', stripped).group()
            content = stripped.strip()

            if buffer:
                if (len(buffer.strip()) > 40 and len(content) > 40 and
                        current_prefix == prefix and
                        not re.search(r'[.!?:]$', buffer.strip())):
                    buffer += " " + content
                    continue
                else:
                    merged.append(prefix + buffer)
                    buffer = content
                    prefix = current_prefix
            else:
                buffer = content
                prefix = current_prefix

        if buffer:
            merged.append(prefix + buffer)

        return merged

    def wrap_line_intelligently(self, line: str, max_length: int = None) -> List[str]:
        """
        Wrap lines intelligently while preserving form structure and avoiding word truncation
        """
        if max_length is None:
            max_length = self.max_line_length

        if len(line) <= max_length:
            return [line]

        # Preserve leading whitespace exactly
        leading_spaces = len(line) - len(line.lstrip())
        content = line[leading_spaces:]
        prefix = ' ' * leading_spaces

        # For form-like content, be more conservative with wrapping
        if leading_spaces > 20 or len(content.strip()) < 15:
            # Still need to wrap if it exceeds max length
            if len(line) <= max_length:
                return [line]

        # Split at natural break points to avoid word truncation
        wrapped_lines = []
        current_line = ""

        # Split content into words while preserving spaces
        words = re.findall(r'\S+|\s+', content)

        for word in words:
            test_line = current_line + word

            if len(prefix + test_line) <= max_length:
                current_line = test_line
            else:
                # If current line is not empty, save it and start new line
                if current_line.strip():
                    wrapped_lines.append(prefix + current_line.rstrip())
                    current_line = word.lstrip() if word.isspace() else word
                else:
                    # Word itself is too long, need to break it
                    if len(word) > max_length - len(prefix):
                        # Break the word at character boundary
                        available_space = max_length - len(prefix)
                        wrapped_lines.append(prefix + word[:available_space])
                        remaining = word[available_space:]
                        current_line = remaining
                    else:
                        current_line = word

        # Add remaining content
        if current_line.strip():
            wrapped_lines.append(prefix + current_line.rstrip())

        return wrapped_lines if wrapped_lines else [line]

    def translate_line(self, line: str, line_number: int = 0) -> List[str]:
        """
        Translate a line while preserving formatting and handling Y/N replacements
        """
        # Preserve empty lines and heavily formatted lines
        if not line.strip() or re.match(r'^[\s\-\=\*\#]+$', line):
            return [line]

        # Check if this is a form header or code line - preserve exactly
        if (re.search(r'(NML\d+|DFA-[A-Z]+-\d+)', line) or
            len(line.strip()) < 5 or
            re.match(r'^[\s\{\}\(\)\[\]]+$', line.strip())):
            return [line]

        # Capture original Y/N positions for alignment
        yn_positions = self._detect_yn_positions(line)

        # Handle Y/N replacements in the entire line first
        line_with_yn = self.handle_yn_replacements(line)

        # Extract translatable segments
        segments = self.extract_translatable_content(line_with_yn)

        if not segments:
            return self.wrap_line_intelligently(line_with_yn)

        # Translate each segment
        result_line = line_with_yn
        offset = 0

        for segment in segments:
            original_text = segment['text']
            translated_text = self.translate_text_with_context(original_text, line_with_yn)

            # Calculate positions with offset
            start_pos = segment['start'] + offset
            end_pos = segment['end'] + offset

            # Replace in result line
            result_line = (result_line[:start_pos] +
                          translated_text +
                          result_line[end_pos:])

            # Update offset
            offset += len(translated_text) - len(original_text)

        # Align S/N indicators
        result_line = self._align_yes_no_spacing(line, result_line, yn_positions)

        # Handle line wrapping to prevent truncation
        wrapped_lines = self.wrap_line_intelligently(result_line)

        return wrapped_lines

    def translate_file(self, input_file_path: str, output_file_path: Optional[str] = None) -> None:
        """
        Translate file with improved handling
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        # Generate output filename
        if output_file_path is None:
            base_name = os.path.splitext(input_file_path)[0]
            extension = os.path.splitext(input_file_path)[1] or '.txt'
            output_file_path = f"{base_name}_spanish{extension}"

        if not output_file_path.endswith('.txt'):
            output_file_path += '.txt'

        self.logger.info(f"Input file: {input_file_path}")
        self.logger.info(f"Output file: {output_file_path}")

        # Read file
        raw_lines = self._read_file_with_encoding(input_file_path)
        lines = self.merge_paragraph_lines(raw_lines)
        total_lines = len(lines)

        self.logger.info(f"Processing {total_lines} lines...")

        # Process lines
        translated_lines = []
        for i, line in enumerate(lines):
            line_content = line.rstrip('\n\r')

            # Translate line or paragraph
            translated_line_list = self.translate_line(line_content, i)

            for translated_line in translated_line_list:
                translated_lines.append(translated_line)

            # Progress update
            if (i + 1) % 5 == 0 or i == total_lines - 1:
                self.logger.info(f"Progress: {i + 1}/{total_lines} lines completed")

        # Write output
        self._write_output_file(output_file_path, translated_lines)

    def _read_file_with_encoding(self, file_path: str) -> List[str]:
        """Read file with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.readlines()
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(f"Could not decode file {file_path}")

    def _write_output_file(self, output_path: str, lines: List[str]) -> None:
        """Write output file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                for line in lines:
                    padded = line[:self.max_line_length].ljust(self.max_line_length)
                    file.write(padded + '\n')

            self.logger.info(f"Translation completed successfully!")
            self.logger.info(f"Output saved to: {output_path}")
            self.logger.info(f"File size: {os.path.getsize(output_path)} bytes")

        except Exception as e:
            self.logger.error(f"Error writing file: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Translate text files using Azure OpenAI")
    parser.add_argument("input_file", help="Input text file path")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--endpoint", required=True, help="Azure OpenAI endpoint URL")
    parser.add_argument("--api-key", required=True, help="Azure OpenAI API key")
    parser.add_argument("--api-version", default="2024-02-01", help="API version")
    parser.add_argument("--deployment", default="gpt-4", help="Deployment name")
    parser.add_argument("--max-line-length", type=int, default=80, help="Maximum line length")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1

    try:
        translator = TextFileTranslator(
            azure_endpoint=args.endpoint,
            api_key=args.api_key,
            api_version=args.api_version,
            deployment_name=args.deployment,
            max_line_length=args.max_line_length
        )

        translator.translate_file(args.input_file, args.output)
        print("Translation completed successfully!")

    except Exception as e:
        print(f"Translation failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
