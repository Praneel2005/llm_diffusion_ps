import fitz
import re
import json
import os

def parse_paper(pdf_path: str) -> dict:
    print(f"\n[PDF Parser] Reading: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    result = {
        "title": _extract_title(full_text),
        "abstract": _extract_abstract(full_text),
        "introduction": _extract_section(full_text, "introduction"),
        "conclusion": _extract_conclusion(full_text),
        "figure_captions": _extract_figure_captions(full_text),
    }

    print(f"\n[PDF Parser] ---- EXTRACTION SUMMARY ----")
    print(f"  Title found:       {result['title'][:60]}...")
    print(f"  Figure captions:   {len(result['figure_captions'])} found")
    return result

def _extract_title(text: str) -> str:
    lines = text.strip().split("\n")
    for line in lines:
        if 10 < len(line.strip()) < 150: return line.strip()
    return "Title not found"

def _extract_abstract(text: str) -> str:
    pattern = re.search(r'(?i)abstract\s*\n(.*?)(?=\n\s*(?:1\.|introduction|keywords))', text, re.DOTALL)
    return re.sub(r'\s+', ' ', pattern.group(1).strip()) if pattern else "Abstract not found"

def _extract_section(text: str, section_name: str) -> str:
    pattern = re.search(rf'(?i)(?:1\.\s*)?{section_name}\s*\n(.*?)(?=\n\s*(?:\d+\.|2\s+\w))', text, re.DOTALL)
    return re.sub(r'\s+', ' ', pattern.group(1).strip())[:2000] if pattern else f"{section_name} not found"

def _extract_conclusion(text: str) -> str:
    pattern = re.search(r'(?i)conclusions?\s*\n(.*?)(?=\n\s*(?:references|acknowledgments|appendix|\d+\.\s*\w))', text, re.DOTALL)
    return re.sub(r'\s+', ' ', pattern.group(1).strip())[:2000] if pattern else "Conclusion not found"

def _extract_figure_captions(text: str) -> list:
    captions = []
    pattern = re.finditer(r'(?i)(?:figure|fig\.?)\s*(\d+)[:\.]?\s*([^\n]{10,500})', text)
    seen_nums = set()
    for match in pattern:
        fig_num = match.group(1)
        caption = re.sub(r'\s+', ' ', match.group(2).strip())
        if fig_num not in seen_nums and len(caption) >= 15:
            seen_nums.add(fig_num)
            captions.append({"figure_num": fig_num, "caption": caption})
    captions.sort(key=lambda x: int(x["figure_num"]))
    return captions

def save_extraction(result: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw_pdfs/test_paper.pdf"
    result = parse_paper(pdf_path)
    save_extraction(result, "data/extracted/test_paper.json")
