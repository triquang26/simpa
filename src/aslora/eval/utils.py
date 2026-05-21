"""GSM8K answer extraction helpers."""
import re


def extract_gsm_num(text: str) -> int:
    pattern = r'####\s*(\d+)'
    match = re.search(pattern, text)
    result = match.group(1) if match else ""
    try:
        return int(result.replace(",", ""))
    except Exception:
        return 0


def extract_answer(text: str) -> int:
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if match:
        return int(float(match.group(1).replace(",", "")))
    nums = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', text)
    if nums:
        try:
            return int(float(nums[-1].replace(",", "")))
        except Exception:
            return 0
    return 0
