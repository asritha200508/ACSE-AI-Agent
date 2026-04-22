import os
import shutil
import tempfile
import pandas as pd
import re

def _safe_copy(filepath: str) -> str:
    """Copy file to a temp path to bypass Windows file locks. Returns temp path."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(filepath, tmp_path)
    return tmp_path


def load_staff(filepath: str) -> pd.DataFrame:
    """
    Load staff data from the active sheet of the Excel workbook using openpyxl.
    Reads the sheet as-is, finds the header row, and extracts:
      Sl.No, E.Code, Name of the Faculty, Designation, Mobile No.
    """
    from openpyxl import load_workbook

    tmp_path = _safe_copy(filepath)
    try:
        wb = load_workbook(tmp_path, data_only=True)
        ws = wb.active  # Always read the active (most recent) sheet
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ── Find the header row ────────────────────────────────────────────────────
    header_row_num = None
    for row in ws.iter_rows():
        vals = [str(c.value).strip().lower() for c in row if c.value]
        joined = " ".join(vals)
        if "name of the faculty" in joined or ("name" in joined and "designation" in joined):
            header_row_num = row[0].row
            break

    if header_row_num is None:
        return pd.DataFrame()

    # ── Read header columns and their positions ────────────────────────────────
    header_row = list(ws.iter_rows(min_row=header_row_num, max_row=header_row_num))[0]
    col_map = {}  # normalized name -> column index (0-based)
    for i, cell in enumerate(header_row):
        raw = str(cell.value or "").strip().lower().replace(".", "").replace(" ", "")
        col_map[raw] = i

    # Determine column indices for the 5 fields we care about
    def _find_col(*keys):
        for k in keys:
            nk = k.lower().replace(".", "").replace(" ", "")
            if nk in col_map:
                return col_map[nk]
            # partial match
            for mapped_key, idx in col_map.items():
                if nk in mapped_key or mapped_key in nk:
                    return idx
        return None

    slno_idx   = _find_col("slno", "sl.no")
    ecode_idx  = _find_col("ecode", "e.code")
    name_idx   = _find_col("nameofthefaculty", "name of the faculty", "name")
    desig_idx  = _find_col("designation", "desig")
    mobile_idx = _find_col("mobileno", "mobile no", "mobile no.", "phone")

    if name_idx is None:
        return pd.DataFrame()

    def _cell_val(cells: list, idx) -> str:
        """Extract a clean string value from a cell by column index."""
        if idx is None or idx >= len(cells):
            return ""
        v = cells[idx].value
        if v is None:
            return ""
        return " ".join(str(v).replace("\n", " ").split()).strip()

    # ── Extract data rows ──────────────────────────────────────────────────────
    records = []
    for row in ws.iter_rows(min_row=header_row_num + 1):
        cells = list(row)

        name = _cell_val(cells, name_idx)
        # CRITICAL: Filter out empty names, "nan", "none", or repeated "Name" headers
        if not name or name.lower() in ("nan", "none", "name of the faculty", "name", "faculty name"):
            continue

        desig  = _cell_val(cells, desig_idx)
        sl_no  = _cell_val(cells, slno_idx)
        ecode  = _cell_val(cells, ecode_idx)
        mobile = _cell_val(cells, mobile_idx)

        # Fallback: scan right-to-left for a phone number
        if not mobile:
            for cell in reversed(cells):
                s = str(cell.value or "").strip().replace(" ", "").replace("-", "")
                if s.isdigit() and 8 <= len(s) <= 12:
                    mobile = s
                    break

        # Clean sl.no (remove decimal point from float-like values)
        try:
            sl_no = str(int(float(sl_no))) if sl_no else ""
        except (ValueError, TypeError):
            pass

        records.append({
            "Sl.No":       sl_no,
            "E.Code":      ecode,
            "Name":        name,
            "Designation": desig,
            "Mobile No":   mobile,
        })

    df = pd.DataFrame(records)
    return df


def load_invigilation(filepath: str) -> pd.DataFrame:
    """Load invigilation data using openpyxl from the active sheet."""
    from openpyxl import load_workbook

    tmp_path = _safe_copy(filepath)
    try:
        wb = load_workbook(tmp_path, data_only=True)
        ws = wb.active
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Find header row
    header_row_num = None
    for row in ws.iter_rows():
        vals = [str(c.value or "").strip().lower() for c in row if c.value]
        joined = " ".join(vals)
        if "room" in joined or "floor" in joined or "faculty" in joined or "invigilator" in joined:
            header_row_num = row[0].row
            break

    if header_row_num is None:
        return pd.DataFrame()

    header_cells = list(ws.iter_rows(min_row=header_row_num, max_row=header_row_num))[0]
    columns = [str(c.value or "").strip() or f"Col{i}" for i, c in enumerate(header_cells)]

    records = []
    for row in ws.iter_rows(min_row=header_row_num + 1):
        vals = [" ".join(str(c.value or "").replace("\n", " ").split()) for c in row]
        if any(v for v in vals):
            records.append(dict(zip(columns, vals)))

    df = pd.DataFrame(records)
    df = df[df.apply(lambda r: any(r.values), axis=1)].reset_index(drop=True)
    return df
