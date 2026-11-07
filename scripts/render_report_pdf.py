"""Render the markdown report to PDF (organisers require a PDF).

    pip install markdown playwright && playwright install chromium     # one time (or just use the browser fallback)
    python scripts/render_report_pdf.py report/Hackathon_Solution_Report.md

Writes <name>.html and <name>.pdf next to the markdown file. If Playwright is not available, open the .html file in
Chrome/Edge and use Print -> Save as PDF (turn on "Background graphics").
It also warns about unreplaced placeholders and about leftover 'Facts' lists written by the AI assistant.
"""
import re
import sys
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; max-width: 190mm; margin: auto; }
h1 { font-size: 20pt; border-bottom: 2px solid #222; padding-bottom: 4px; } h2 { font-size: 14pt; margin-top: 18px; border-bottom: 1px solid #aaa; }
h3 { font-size: 11.5pt; } table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 8px 0; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3px 5px; } th { background: #eee; } img { max-width: 100%; page-break-inside: avoid; }
code { background: #f1f1f1; padding: 0 3px; font-size: 9pt; } pre { background: #f6f6f6; padding: 8px; font-size: 8.8pt; overflow-x: auto; }
blockquote { border-left: 3px solid #888; margin-left: 0; padding-left: 10px; color: #444; }
"""


def browser_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Print to PDF with an installed Edge/Chrome/Chromium in headless mode (no extra Python packages needed)."""
    import os
    import shutil
    import subprocess
    cands = [shutil.which(n) for n in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser")]
    for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if base:
            cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    cands += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/opt/pw-browsers/chromium"]
    for exe in [c for c in cands if c and os.path.isfile(c)]:
        for flags in (["--headless=new"], ["--headless"]):
            cmd = [exe, *flags, "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
            try:
                subprocess.run(cmd, capture_output=True, timeout=120)
            except Exception:
                continue
            if pdf_path.exists() and pdf_path.stat().st_size > 5000:
                return True
    return False


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "report/Hackathon_Solution_Report.md").resolve()
    text = src.read_text(encoding="utf-8")
    todo = len(re.findall(r"TEAM TO (WRITE|FILL)|\[TEAM|UPDATE to the final", text))
    facts = len(re.findall(r"\*Facts", text))
    if todo or facts:
        print(f"WARNING: {todo} placeholder(s) and {facts} AI-written 'Facts' list(s) are still in the report. Replace them with your own text before submitting.")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Technical Solution Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    out_html = src.with_suffix(".html")
    out_html.write_text(html, encoding="utf-8")
    pdf = src.with_suffix(".pdf")
    if browser_pdf(out_html, pdf):
        print("wrote", pdf, "(via headless browser)")
        return
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page()
            page.goto(out_html.as_uri())
            page.wait_for_timeout(800)
            page.pdf(path=str(src.with_suffix(".pdf")), format="A4", print_background=True,
                     margin=dict(top="16mm", bottom="16mm", left="14mm", right="14mm"))
            b.close()
        print("wrote", src.with_suffix(".pdf"))
    except Exception as e:
        print(f"Playwright PDF failed ({e}).\nOpen {out_html} in Chrome or Edge, then Print -> Save as PDF (enable Background graphics).")


if __name__ == "__main__":
    main()
