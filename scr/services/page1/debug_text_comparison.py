# Mục đích: dùng chung quy tắc so sánh kết quả VietOCR và PP-OCRv6.
# Chức năng: ghép token không dấu, nhận diện cặp dễ nhầm và sửa token theo PP-OCRv6.
# Luồng xử lý: hai chuỗi OCR → token cùng vị trí → lọc dấu → so sánh hoặc thay thế.
# Cấu hình/parameter: danh sách cặp ký tự gần giống được khai báo trong module.
# Quan hệ: được module trích xuất field và visualization của page 1 sử dụng.
# Input/output: nhận chuỗi hoặc text region; trả cặp token, sai khác hoặc chuỗi đã sửa.
# Contract: chỉ PP-OCRv6 được phép thay token không dấu trong value đã trích xuất.

import re
import unicodedata


TOKEN_PATTERN = re.compile(
    r"(?:[^\W_]|[+&`])+",
    re.UNICODE,
)
CONFUSABLE_TEXT_PAIRS = (
    ("m", "rn"),
    ("m", "nn"),
    ("w", "vv"),
    ("I", "l"),
    ("I", "1"),
    ("I", "`"),
    ("1", "l"),
    ("1", "7"),
    ("1", "T"),
    ("4", "+"),
    ("0", "O"),
    ("0", "o"),
    ("0", "D"),
    ("O", "Q"),
    ("O", "D"),
    ("o", "a"),
    ("o", "e"),
    ("5", "S"),
    ("5", "s"),
    ("S", "8"),
    ("8", "&"),
    ("2", "Z"),
    ("2", "z"),
    ("Z", "7"),
    ("8", "B"),
    ("8", "3"),
    ("B", "R"),
    ("6", "G"),
    ("6", "b"),
    ("G", "C"),
    ("9", "g"),
    ("9", "q"),
    ("q", "g"),
    ("c", "e"),
    ("c", "o"),
    ("e", "o"),
    ("a", "e"),
    ("w", "v"),
    ("u", "v"),
    ("v", "y"),
    ("h", "b"),
    ("h", "n"),
    ("b", "6"),
    ("n", "r"),
    ("r", "v"),
    ("f", "t"),
    ("t", "l"),
)
CONFUSABLE_TRANSFORMS = tuple(
    transform
    for pair in CONFUSABLE_TEXT_PAIRS
    for transform in (pair, tuple(reversed(pair)))
)


def has_vietnamese_mark(token):
    """Kiểm tra token có chữ mang dấu hoặc chữ đ hay không."""
    if "đ" in token.lower():
        return True

    normalized_token = unicodedata.normalize("NFD", token)
    return any(
        unicodedata.combining(character)
        for character in normalized_token
    )


def pair_non_vietnamese_tokens(vietocr_text, ppocr_text):
    """Ghép token không dấu của hai model theo vị trí trong cùng crop."""
    vietocr_tokens = TOKEN_PATTERN.findall(vietocr_text)
    ppocr_tokens = TOKEN_PATTERN.findall(ppocr_text)
    token_pairs = []

    for token_index, vietocr_token in enumerate(vietocr_tokens):
        if token_index >= len(ppocr_tokens):
            continue

        ppocr_token = ppocr_tokens[token_index]
        if has_vietnamese_mark(vietocr_token):
            continue
        if has_vietnamese_mark(ppocr_token):
            continue
        token_pairs.append((vietocr_token, ppocr_token))

    return token_pairs


def find_confusable_changes(source_text, target_text):
    """Trả các phép đổi ký tự gần giống nếu chúng giải thích toàn bộ sai khác."""
    compared_states = {}

    def compare(source_index, target_index):
        state = (source_index, target_index)
        if state in compared_states:
            return compared_states[state]
        if source_index == len(source_text) and target_index == len(
            target_text
        ):
            return []

        if (
            source_index < len(source_text)
            and target_index < len(target_text)
            and source_text[source_index] == target_text[target_index]
        ):
            remaining_changes = compare(
                source_index + 1,
                target_index + 1,
            )
            if remaining_changes is not None:
                compared_states[state] = remaining_changes
                return remaining_changes

        for source, target in CONFUSABLE_TRANSFORMS:
            if not source_text.startswith(source, source_index):
                continue
            if not target_text.startswith(target, target_index):
                continue

            remaining_changes = compare(
                source_index + len(source),
                target_index + len(target),
            )
            if remaining_changes is not None:
                changes = [(source, target), *remaining_changes]
                compared_states[state] = changes
                return changes

        compared_states[state] = None
        return None

    changes = compare(0, 0)
    return changes or None


def correct_value_with_ppocr(value, text_regions):
    """Thay token không dấu trong field value bằng kết quả PP-OCRv6 khác biệt."""
    corrections = {}
    for region in text_regions:
        token_pairs = pair_non_vietnamese_tokens(
            region["text"],
            region["ppocr_text"],
        )
        for vietocr_token, ppocr_token in token_pairs:
            if vietocr_token == ppocr_token:
                continue
            if find_confusable_changes(vietocr_token, ppocr_token) is None:
                continue
            corrections.setdefault(vietocr_token, []).append(ppocr_token)

    corrected_parts = []
    previous_end = 0
    for match in TOKEN_PATTERN.finditer(value):
        token = match.group(0)
        corrected_parts.append(value[previous_end:match.start()])
        token_corrections = corrections.get(token)
        if token_corrections:
            corrected_parts.append(token_corrections.pop(0))
        else:
            corrected_parts.append(token)
        previous_end = match.end()

    corrected_parts.append(value[previous_end:])
    return "".join(corrected_parts)
