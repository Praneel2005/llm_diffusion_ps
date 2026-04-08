"""
shared/pdf_parser.py — Layer 2 version with table filtering
"""
import os, requests, json
import xml.etree.ElementTree as ET

GROBID_URL = "http://localhost:8070"
NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

def _get_text(element):
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()

def _is_table(fig_el):
    if fig_el.attrib.get('type', '').lower() == 'table':
        return True
    head = fig_el.find('tei:head', NS)
    if head is not None and head.text:
        if head.text.strip().lower().startswith('table'):
            return True
    return False

def parse_pdf(pdf_path):
    print(f"\n[Parser] Sending to GROBID: {os.path.basename(pdf_path)}")
    with open(pdf_path, 'rb') as f:
        response = requests.post(
            f"{GROBID_URL}/api/processFulltextDocument",
            files={'input': f},
            data={'consolidateHeader': '1'},
            timeout=120
        )
    if response.status_code != 200:
        raise RuntimeError(f"GROBID failed: {response.status_code}")

    root = ET.fromstring(response.text)
    abstract, introduction, conclusion = "", "", ""

    el = root.find('.//tei:abstract', NS)
    if el is not None:
        abstract = _get_text(el)

    for div in root.findall('.//tei:body//tei:div', NS):
        head = div.find('tei:head', NS)
        if head is None or not head.text:
            continue
        h = head.text.lower().strip()
        if "introduction" in h and not introduction:
            introduction = _get_text(div)[:3000]
        elif any(w in h for w in ["conclusion","summary","discussion"]) and not conclusion:
            conclusion = _get_text(div)[:2000]

    figures, tables_skipped = [], 0
    for fig_el in root.findall('.//tei:figure', NS):
        if _is_table(fig_el):
            tables_skipped += 1
            continue
        fig_id = fig_el.attrib.get('{http://www.w3.org/XML/1998/namespace}id', f'fig_{len(figures)}')
        head = fig_el.find('tei:head', NS)
        desc = fig_el.find('tei:figDesc', NS)
        parts = []
        if head is not None and head.text:
            parts.append(head.text.strip())
        if desc is not None:
            parts.append(_get_text(desc))
        caption = " ".join(parts).strip()
        if len(caption) >= 20:
            figures.append({"id": fig_id, "caption": caption})

    if not figures:
        print("[Parser] WARNING: 0 figures found after filtering.")

    result = {
        "abstract": abstract,
        "introduction": introduction,
        "conclusion": conclusion,
        "figures": figures,
        "tables_skipped": tables_skipped
    }
    return result

def save_extracted(result, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Parser] Saved to: {output_path}")

def print_summary(result):
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Abstract length  : {len(result['abstract'])} chars")
    print(f"Intro length     : {len(result['introduction'])} chars")
    print(f"Conclusion length: {len(result['conclusion'])} chars")
    print(f"Figures found    : {len(result['figures'])}")
    print(f"Tables skipped   : {result['tables_skipped']}")
    print("\nFigures (no tables):")
    for fig in result['figures']:
        print(f"  [{fig['id']}] {fig['caption'][:90]}...")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python shared/pdf_parser.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)
    result = parse_pdf(pdf_path)
    base = os.path.basename(pdf_path).replace(".pdf", "_extracted.json")
    out_path = os.path.join("data", "extracted", base)
    save_extracted(result, out_path)
    print_summary(result)
