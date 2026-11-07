"""Convert the markdown report to .docx with pandoc (images and tables included).

    python scripts/render_report_docx.py report/Hackathon_Solution_Report.md
Needs pandoc (https://pandoc.org/installing.html) or `pip install pypandoc_binary`.
"""
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "report/Hackathon_Solution_Report.md").resolve()
    out = src.with_suffix(".docx")
    exe = shutil.which("pandoc")
    if exe is None:
        try:
            import pypandoc
            exe = pypandoc.get_pandoc_path()
        except Exception:
            sys.exit("pandoc not found: install it from https://pandoc.org/installing.html or run `pip install pypandoc_binary`")
    subprocess.run([exe, str(src), "-f", "gfm", "-o", str(out), f"--resource-path={src.parent}", "--columns=100"], check=True)
    polish(out)
    print("wrote", out)


def polish(path):
    """Grid borders and compact font for tables (pandoc's default tables are borderless)."""
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt
    from docx.shared import Inches
    doc = Document(str(path))
    for t in doc.tables:
        n = len(t.columns)
        first = 1.6 if n == 2 else (1.9 if n >= 7 else 1.5)
        widths = [first] + [(6.5 - first) / (n - 1)] * (n - 1)
        t.autofit = False
        for row in t.rows:
            for w, cell in zip(widths, row.cells):
                cell.width = Inches(w)
        tblPr = t._tbl.tblPr
        tblW = tblPr.find(qn("w:tblW"))
        if tblW is None:
            tblW = OxmlElement("w:tblW")
            tblPr.append(tblW)
        tblW.set(qn("w:type"), "dxa")
        tblW.set(qn("w:w"), str(int(6.5 * 1440)))
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "0"), ("w:color", "808080")):
                el.set(qn(k), v)
            borders.append(el)
        tblPr.append(borders)
        for row in t.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(8)
        for cell in t.rows[0].cells:                      # shade the header row
            tcPr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "E7E6E6")
            tcPr.append(shd)
    from docx.shared import RGBColor
    wanted = {"title", "heading 1", "heading 2", "heading 3", "heading 4", "subtitle"}
    for st in doc.styles:                       # pandoc's default theme uses blue headings; match names case-insensitively
        if st.name and st.name.lower() in wanted:
            rpr = st.element.get_or_add_rPr()
            for el in rpr.findall(qn("w:color")):
                rpr.remove(el)
            st.font.color.rgb = RGBColor(0, 0, 0)
    doc.save(str(path))


if __name__ == "__main__":
    main()
