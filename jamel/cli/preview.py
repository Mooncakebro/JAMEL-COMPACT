"""Parquet preview web server."""

from __future__ import annotations

import base64
import html
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency should exist
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - dependency should exist
    torch = None  # type: ignore[assignment]


@dataclass
class RenderConfig:
    max_text_len: int = 200
    max_collection_len: int = 2000
    image_thumb_max_side: int = 256
    image_full_max_side: int = 1024


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _encode_image(img: Image.Image, max_side: int) -> tuple[str, str]:
    img = img.convert("RGBA")
    if max_side > 0:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", f"{img.width}x{img.height}"


def _image_from_bytes(data: bytes, max_side: int) -> tuple[str, str] | None:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return _encode_image(img, max_side)
    except Exception:
        return None


def _normalize_array(arr: "np.ndarray") -> "np.ndarray":
    if arr.dtype == np.uint8:
        return arr
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    if arr_max <= 1.0 and arr_min >= 0.0:
        scaled = arr * 255.0
    else:
        scaled = (arr - arr_min) / (arr_max - arr_min) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _image_from_array(arr: "np.ndarray", max_side: int) -> tuple[str, str] | None:
    if np is None:
        return None
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] not in (1, 3, 4):
        return None
    arr = _normalize_array(arr)
    try:
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        img = Image.fromarray(arr, mode="RGB" if arr.shape[-1] == 3 else "RGBA")
        return _encode_image(img, max_side)
    except Exception:
        return None


def _render_text_modal_trigger(
    title: str,
    content: str,
    label: str,
    row_index: int | None = None,
    column_name: str | None = None,
) -> str:
    if row_index is not None and column_name is not None:
        content_attrs = (
            f"data-row-index='{row_index}' "
            f"data-column='{_escape(column_name)}'"
        )
    else:
        content_attrs = f"data-content='{_escape(content)}'"

    return (
        "<button class='text-trigger' type='button' "
        f"data-title='{_escape(title)}' {content_attrs}>"
        f"<span class='text-trigger-label'>{_escape(label)}</span>"
        "</button>"
    )


def _preview_label(prefix: str, content: str, limit: int) -> str:
    compact = " ".join(content.split())
    snippet = _shorten(compact, limit)
    if snippet:
        return f"{prefix}: {snippet}"
    return prefix


def _format_full_text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()

    if np is not None and isinstance(value, np.ndarray):
        return np.array2string(value, threshold=20)

    if torch is not None and isinstance(value, torch.Tensor):
        return str(value.detach().cpu().numpy())

    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

    return str(value)


def _render_value(
    value: Any,
    cfg: RenderConfig,
    row_index: int | None = None,
    column_name: str | None = None,
) -> str:
    if value is None:
        return "<span class='empty'>None</span>"

    if isinstance(value, float) and pd.isna(value):
        return "<span class='empty'>NaN</span>"

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        thumb = _image_from_bytes(raw, cfg.image_thumb_max_side)
        full = _image_from_bytes(raw, cfg.image_full_max_side)
        if thumb is not None:
            thumb_src, thumb_size = thumb
            full_src = full[0] if full is not None else thumb_src
            return (
                "<div class='img-cell'>"
                f"<img class='zoomable' src='{thumb_src}' data-full='{full_src}' alt='image'/>"
                f"<div class='meta'>{thumb_size}</div>"
                "</div>"
            )
        label = f"bytes ({len(raw)} bytes)"
        summary = _preview_label(label, raw.hex(), cfg.max_text_len)
        return _render_text_modal_trigger(
            label,
            raw.hex(),
            summary,
            row_index,
            column_name,
        )

    if np is not None and isinstance(value, np.ndarray):
        thumb = _image_from_array(value, cfg.image_thumb_max_side)
        full = _image_from_array(value, cfg.image_full_max_side)
        if thumb is not None:
            thumb_src, thumb_size = thumb
            full_src = full[0] if full is not None else thumb_src
            return (
                "<div class='img-cell'>"
                f"<img class='zoomable' src='{thumb_src}' data-full='{full_src}' alt='ndarray'/>"
                f"<div class='meta'>{thumb_size}</div>"
                "</div>"
            )
        summary = f"ndarray shape={value.shape} dtype={value.dtype}"
        full_text = _format_full_text(value)
        label = _preview_label(summary, full_text, cfg.max_text_len)
        return _render_text_modal_trigger(summary, full_text, label, row_index, column_name)

    if torch is not None and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        thumb = _image_from_array(arr, cfg.image_thumb_max_side) if np is not None else None
        full = _image_from_array(arr, cfg.image_full_max_side) if np is not None else None
        if thumb is not None:
            thumb_src, thumb_size = thumb
            full_src = full[0] if full is not None else thumb_src
            return (
                "<div class='img-cell'>"
                f"<img class='zoomable' src='{thumb_src}' data-full='{full_src}' alt='tensor'/>"
                f"<div class='meta'>{thumb_size}</div>"
                "</div>"
            )
        summary = f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
        full_text = _format_full_text(value)
        label = _preview_label(summary, full_text, cfg.max_text_len)
        return _render_text_modal_trigger(summary, full_text, label, row_index, column_name)

    if isinstance(value, (dict, list, tuple)):
        full_text = _format_full_text(value)
        label = _preview_label(type(value).__name__, full_text, cfg.max_text_len)
        return _render_text_modal_trigger(
            type(value).__name__,
            full_text,
            label,
            row_index,
            column_name,
        )

    if isinstance(value, str):
        inline_limit = min(cfg.max_text_len, 48)
        compact = " ".join(value.split())
        if value:
            label = compact if len(compact) <= inline_limit else _preview_label("text", value, inline_limit)
            return _render_text_modal_trigger(
                "text",
                value,
                label,
                row_index,
                column_name,
            )
        return "<pre></pre>"

    return f"<pre>{_escape(str(value))}</pre>"


def _resolve_column_order(
    df: pd.DataFrame,
    requested_columns: Iterable[str] | None,
) -> list[Any]:
    columns = list(df.columns)
    if not requested_columns:
        return columns

    by_name = {str(col): col for col in columns}
    ordered = []
    seen = set()
    for name in requested_columns:
        if name in by_name and name not in seen:
            ordered.append(by_name[name])
            seen.add(name)

    ordered.extend(col for col in columns if str(col) not in seen)
    return ordered


def _render_rows(
    df: pd.DataFrame,
    start: int,
    end: int,
    cfg: RenderConfig,
    columns: Iterable[str] | None = None,
) -> list[str]:
    ordered_columns = _resolve_column_order(df, columns)
    rows = []
    for row_offset, (_, row) in enumerate(
        df.loc[:, ordered_columns].iloc[start:end].iterrows()
    ):
        row_index = start + row_offset
        cells = []
        for column, value in zip(ordered_columns, row.values.tolist()):
            cells.append(
                f"<td>{_render_value(value, cfg, row_index, str(column))}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return rows


def _render_schema(df: pd.DataFrame) -> str:
    items = []
    for col, dtype in df.dtypes.items():
        items.append(
            "<li>"
            f"<span class='col'>{_escape(str(col))}</span>"
            f"<span class='dtype'>{_escape(str(dtype))}</span>"
            "</li>"
        )
    return "<ul class='schema'>" + "".join(items) + "</ul>"


def _load_parquet(path: str, columns: Iterable[str] | None) -> pd.DataFrame:
    if columns:
        return pd.read_parquet(path, columns=list(columns))
    return pd.read_parquet(path)


def _build_file_list(parquet_paths: list[str]) -> list[dict[str, str]]:
    seen = {}
    normalized_paths = [os.path.abspath(path) for path in parquet_paths]
    common_root = os.path.commonpath(normalized_paths) if normalized_paths else ""
    items = []
    for idx, path in enumerate(normalized_paths):
        name = os.path.basename(path)
        count = seen.get(name, 0)
        seen[name] = count + 1
        display = name if count == 0 else f"{name} ({count + 1})"
        relpath = os.path.relpath(path, common_root) if common_root else path
        items.append(
            {
                "id": str(idx),
                "name": display,
                "basename": name,
                "path": path,
                "relpath": relpath,
            }
        )
    return items


def _render_index(title: str, page_size: int) -> str:
    return (
        "<!DOCTYPE html>"
        "<html lang='en'>"
        "<head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        f"<title>{_escape(title)}</title>"
        "<style>"
        ":root{--paper:#f7f4ef;--panel:#fbf7f1;--panel-strong:#f2e9dd;--line:#e2d9cc;--ink:#222;"
        "--muted:#6a5d50;--accent:#9d5f2f;}"
        "body{font-family:'IBM Plex Sans','Segoe UI',sans-serif;background:linear-gradient(135deg,#f7f4ef 0%,#f0e7da 100%);"
        "color:var(--ink);margin:0;}"
        ".wrap{max-width:1600px;margin:24px auto;padding:0 16px;}"
        "h1{font-weight:700;letter-spacing:0.5px;margin:0 0 8px 0;}"
        "h2{margin:18px 0 10px 0;}"
        ".meta{color:var(--muted);font-size:12px;margin-top:4px;}"
        ".layout{display:grid;grid-template-columns:minmax(280px,360px) minmax(0,1fr);gap:18px;align-items:start;}"
        ".sidebar,.content{background:rgba(251,247,241,0.95);border:1px solid var(--line);box-shadow:0 8px 24px rgba(94,72,52,0.08);}"
        ".sidebar{position:sticky;top:16px;max-height:calc(100vh - 32px);display:flex;flex-direction:column;overflow:hidden;}"
        ".sidebar-head,.content-head{padding:16px 18px;border-bottom:1px solid var(--line);background:var(--panel-strong);}"
        ".sidebar-body,.content-body{padding:16px 18px;}"
        ".controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:12px 0 16px 0;}"
        ".controls button{background:var(--panel-strong);border:1px solid var(--line);padding:6px 10px;cursor:pointer;font-weight:600;}"
        ".controls input,.controls select{border:1px solid var(--line);padding:6px 8px;background:#fff;}"
        ".filter-grid{display:grid;grid-template-columns:minmax(0,1fr) 112px;gap:10px;align-items:center;}"
        ".filter-input,.filter-select{width:100%;box-sizing:border-box;border:1px solid var(--line);padding:10px 12px;background:#fff;}"
        ".filter-help{margin:10px 0 12px 0;font-size:12px;color:var(--muted);}"
        ".file-list{display:flex;flex-direction:column;gap:8px;max-height:calc(100vh - 260px);overflow:auto;padding-right:4px;}"
        ".file-item{width:100%;text-align:left;border:1px solid var(--line);background:#fff;padding:10px 12px;cursor:pointer;}"
        ".file-item:hover{border-color:#c8b59e;background:#fcfaf6;}"
        ".file-item.active{background:#f3e7d7;border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent);}"
        ".file-name{display:block;font-weight:700;font-size:13px;line-height:1.35;word-break:break-word;}"
        ".file-path{display:block;margin-top:6px;font-size:12px;line-height:1.4;color:var(--muted);word-break:break-all;}"
        ".file-count{font-size:12px;color:var(--muted);margin-top:10px;}"
        ".selected-file{padding:12px 14px;border:1px solid var(--line);background:#fff;margin:0 0 14px 0;}"
        ".selected-file .selected-name{font-weight:700;display:block;}"
        ".selected-file .selected-path{display:block;margin-top:6px;font-family:'IBM Plex Mono',monospace;font-size:12px;"
        "line-height:1.45;word-break:break-all;color:var(--muted);}"
        ".column-panel{padding:12px 14px;border:1px solid var(--line);background:#fff;margin:0 0 14px 0;}"
        ".column-panel-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px;}"
        ".column-panel-title{font-size:13px;font-weight:700;}"
        ".column-reset{border:1px solid var(--line);background:var(--panel-strong);padding:5px 8px;cursor:pointer;font-size:12px;}"
        ".column-list{display:flex;flex-wrap:wrap;gap:8px;max-height:118px;overflow:auto;}"
        ".column-item{display:flex;align-items:center;gap:6px;border:1px solid var(--line);background:#fcfaf6;"
        "padding:6px 8px;cursor:grab;max-width:240px;}"
        ".column-item.dragging,.column-header.dragging{opacity:0.45;}"
        ".column-grip{color:var(--muted);font-size:13px;line-height:1;}"
        ".column-name{font-size:12px;line-height:1.25;word-break:break-word;}"
        ".schema{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
        "gap:8px;padding:0;list-style:none;margin:16px 0;}"
        ".schema li{background:#fff;border:1px solid var(--line);padding:8px 10px;"
        "display:flex;justify-content:space-between;gap:8px;}"
        ".schema .col{font-weight:600;}"
        ".schema .dtype{color:var(--muted);}"
        "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);}"
        "th,td{border:1px solid var(--line);padding:8px;vertical-align:top;}"
        "th{background:var(--panel-strong);text-align:left;position:sticky;top:0;z-index:1;}"
        ".column-header{cursor:grab;user-select:none;}"
        ".column-header:hover{background:#ead8c3;}"
        "pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:'IBM Plex Mono',monospace;font-size:12px;}"
        ".img-cell{display:flex;flex-direction:column;align-items:flex-start;gap:4px;}"
        ".img-cell img{max-width:100%;border:1px solid var(--line);background:#fff;cursor:zoom-in;}"
        ".empty{color:#9c8f80;font-style:italic;}"
        ".text-trigger{display:block;width:100%;background:#f9f5ef;border:1px solid var(--line);"
        "padding:8px;text-align:left;cursor:pointer;min-width:120px;max-width:220px;}"
        ".text-trigger:hover{background:#f3ebdf;}"
        ".text-trigger-label{display:block;font-size:12px;line-height:1.35;white-space:normal;"
        "word-break:break-word;color:#5f5347;pointer-events:none;max-height:2.7em;overflow:hidden;}"
        ".modal{position:fixed;inset:0;background:rgba(15,12,8,0.65);display:none;"
        "align-items:center;justify-content:center;padding:24px;z-index:10;}"
        ".modal img{max-width:95vw;max-height:95vh;border:2px solid var(--panel-strong);background:#fff;}"
        ".modal-panel{width:min(1100px,95vw);max-height:90vh;background:var(--panel);border:2px solid var(--line);"
        "display:flex;flex-direction:column;}"
        ".modal-head{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;"
        "border-bottom:1px solid var(--line);background:var(--panel-strong);gap:12px;}"
        ".modal-head h3{margin:0;font-size:16px;}"
        ".modal-close{border:1px solid #ccbfae;background:#fff;padding:6px 10px;cursor:pointer;}"
        ".modal-body{padding:16px;overflow:auto;}"
        ".modal-body pre{font-size:13px;line-height:1.45;}"
        "@media (max-width: 980px){.layout{grid-template-columns:1fr;}.sidebar{position:static;max-height:none;}"
        ".file-list{max-height:320px;}}"
        "</style>"
        "</head>"
        "<body>"
        "<div class='wrap'>"
        f"<h1>{_escape(title)}</h1>"
        "<div class='layout'>"
        "<aside class='sidebar'>"
        "<div class='sidebar-head'>"
        "<div class='filter-grid'>"
        "<input id='file-filter' class='filter-input' type='text' placeholder='Search path, filename, directory'/>"
        "<select id='filter-mode' class='filter-select'>"
        "<option value='contains'>Contains</option>"
        "<option value='regex'>Regex</option>"
        "</select>"
        "</div>"
        "<div class='filter-help'>Search against full parquet path. Regex uses JavaScript syntax.</div>"
        "<div class='file-count' id='file-count'></div>"
        "</div>"
        "<div class='sidebar-body'>"
        "<div class='file-list' id='file-list'></div>"
        "</div>"
        "</aside>"
        "<main class='content'>"
        "<div class='content-head'>"
        "<div class='selected-file'>"
        "<span class='selected-name' id='selected-file-name'></span>"
        "<span class='selected-path' id='selected-file-path'></span>"
        "</div>"
        "<div class='column-panel'>"
        "<div class='column-panel-head'>"
        "<span class='column-panel-title'>Column order</span>"
        "<button class='column-reset' type='button' id='column-reset'>Reset</button>"
        "</div>"
        "<div class='column-list' id='column-list'></div>"
        "</div>"
        "<div class='controls'>"
        "<button type='button' id='prev-btn'>Prev</button>"
        "<button type='button' id='next-btn'>Next</button>"
        "<span class='meta' id='page-status'></span>"
        "<label>Page size "
        "<select id='page-size'>"
        "<option value='1'>1</option>"
        "<option value='5'>5</option>"
        "<option value='10'>10</option>"
        "<option value='25'>25</option>"
        "<option value='50'>50</option>"
        "<option value='100'>100</option>"
        "</select>"
        "</label>"
        "<label>Go to "
        "<input id='page-input' type='number' min='1' value='1' style='width:80px'/>"
        "</label>"
        "</div>"
        "<div class='meta' id='file-status'></div>"
        "</div>"
        "<div class='content-body'>"
        "<h2>Schema</h2>"
        "<div id='schema-wrap'></div>"
        "<h2>Preview</h2>"
        "<table>"
        "<thead id='table-head'></thead>"
        "<tbody id='preview-body'></tbody>"
        "</table>"
        "</div>"
        "</main>"
        "</div>"
        "<div class='modal' id='img-modal'>"
        "<img id='modal-img' alt='full'/>"
        "</div>"
        "<div class='modal' id='text-modal'>"
        "<div class='modal-panel'>"
        "<div class='modal-head'>"
        "<h3 id='text-modal-title'></h3>"
        "<button class='modal-close' type='button' id='text-modal-close'>Close</button>"
        "</div>"
        "<div class='modal-body'>"
        "<pre id='text-modal-content'></pre>"
        "</div>"
        "</div>"
        "</div>"
        "<script>"
        f"let pageSize = {page_size};"
        "let totalRows = 0;"
        "let allFiles = [];"
        "let filteredFiles = [];"
        "let currentColumns = [];"
        "let columnOrder = [];"
        "let draggedColumnIndex = null;"
        "const COLUMN_ORDER_STORAGE_KEY = 'parquet-preview-column-order';"
        "const sizeSelect = document.getElementById('page-size');"
        "const pageInput = document.getElementById('page-input');"
        "const pageStatus = document.getElementById('page-status');"
        "const fileStatus = document.getElementById('file-status');"
        "const fileCount = document.getElementById('file-count');"
        "const bodyEl = document.getElementById('preview-body');"
        "const headEl = document.getElementById('table-head');"
        "const schemaWrap = document.getElementById('schema-wrap');"
        "const fileList = document.getElementById('file-list');"
        "const fileFilter = document.getElementById('file-filter');"
        "const filterMode = document.getElementById('filter-mode');"
        "const selectedFileName = document.getElementById('selected-file-name');"
        "const selectedFilePath = document.getElementById('selected-file-path');"
        "const columnList = document.getElementById('column-list');"
        "const columnReset = document.getElementById('column-reset');"
        "let page = 1;"
        "let currentFile = null;"
        "function totalPages(){return Math.max(1, Math.ceil(totalRows / pageSize));}"
        "function clampPage(p){return Math.min(totalPages(), Math.max(1, p));}"
        "function getFileById(fileId){return allFiles.find((item)=>item.id === fileId) || null;}"
        "function escapeHtml(value){"
        "return String(value).replace(/[&<>'\"]/g, (char)=>({"
        "'&':'&amp;','<':'&lt;','>':'&gt;',\"'\":'&#39;','\"':'&quot;'"
        "}[char]));"
        "}"
        "function readSavedColumnOrder(){"
        "try { return JSON.parse(localStorage.getItem(COLUMN_ORDER_STORAGE_KEY) || '[]'); }"
        "catch (error) { return []; }"
        "}"
        "function saveColumnOrder(){"
        "const previous = readSavedColumnOrder().filter((column)=>!columnOrder.includes(column));"
        "localStorage.setItem(COLUMN_ORDER_STORAGE_KEY, JSON.stringify(columnOrder.concat(previous)));"
        "}"
        "function mergeColumnOrder(savedOrder, columns){"
        "const available = new Set(columns);"
        "const ordered = savedOrder.filter((column)=>available.has(column));"
        "columns.forEach((column)=>{ if (!ordered.includes(column)) ordered.push(column); });"
        "return ordered;"
        "}"
        "function moveColumn(fromIndex, toIndex){"
        "if (fromIndex === null || Number.isNaN(fromIndex) || Number.isNaN(toIndex) || fromIndex === toIndex) return;"
        "const nextOrder = columnOrder.slice();"
        "const [column] = nextOrder.splice(fromIndex, 1);"
        "nextOrder.splice(toIndex, 0, column);"
        "columnOrder = nextOrder;"
        "saveColumnOrder();"
        "renderColumnList();"
        "loadPage();"
        "}"
        "function wireColumnDrag(selector){"
        "document.querySelectorAll(selector).forEach((item)=>{"
        "item.addEventListener('dragstart', (event)=>{"
        "draggedColumnIndex = parseInt(item.dataset.columnIndex || '-1', 10);"
        "event.dataTransfer.effectAllowed = 'move';"
        "item.classList.add('dragging');"
        "});"
        "item.addEventListener('dragend', ()=>{"
        "draggedColumnIndex = null;"
        "item.classList.remove('dragging');"
        "});"
        "item.addEventListener('dragover', (event)=>{"
        "event.preventDefault();"
        "event.dataTransfer.dropEffect = 'move';"
        "});"
        "item.addEventListener('drop', (event)=>{"
        "event.preventDefault();"
        "moveColumn(draggedColumnIndex, parseInt(item.dataset.columnIndex || '-1', 10));"
        "});"
        "});"
        "}"
        "function renderColumnList(){"
        "columnList.innerHTML = columnOrder.map((column, index)=>`"
        "<div class='column-item' draggable='true' data-column-index='${index}'>"
        "<span class='column-grip'>::</span>"
        "<span class='column-name'>${escapeHtml(column)}</span>"
        "</div>`).join('');"
        "wireColumnDrag('.column-item');"
        "}"
        "function renderTableHeader(columns){"
        "headEl.innerHTML = `<tr>${columns.map((column, index)=>`"
        "<th class='column-header' draggable='true' data-column-index='${index}'>${escapeHtml(column)}</th>`).join('')}</tr>`;"
        "wireColumnDrag('.column-header');"
        "}"
        "function renderFileList(){"
        "fileList.innerHTML = filteredFiles.map((file)=>{"
        "const active = file.id === currentFile ? ' active' : '';"
        "return `<button type='button' class='file-item${active}' data-file-id='${file.id}'>"
        "<span class='file-name'>${escapeHtml(file.name)}</span>"
        "<span class='file-path'>${escapeHtml(file.path)}</span>"
        "</button>`;"
        "}).join('');"
        "fileCount.textContent = `${filteredFiles.length} / ${allFiles.length} parquet files`;"
        "document.querySelectorAll('.file-item').forEach((btn)=>{"
        "btn.addEventListener('click', async ()=>{"
        "const nextFile = btn.dataset.fileId;"
        "if (!nextFile || nextFile === currentFile) return;"
        "currentFile = nextFile;"
        "page = 1;"
        "await loadMeta();"
        "await loadPage();"
        "renderFileList();"
        "});"
        "});"
        "}"
        "function applyFileFilter(){"
        "const query = fileFilter.value || '';"
        "const mode = filterMode.value || 'contains';"
        "let matcher = null;"
        "if (mode === 'regex' && query) {"
        "try { matcher = new RegExp(query, 'i'); fileFilter.setCustomValidity(''); }"
        "catch (error) { filteredFiles = []; fileFilter.setCustomValidity(error.message); fileFilter.reportValidity(); renderFileList(); return; }"
        "} else { fileFilter.setCustomValidity(''); }"
        "filteredFiles = allFiles.filter((file)=>{"
        "const haystack = [file.name, file.basename, file.relpath, file.path].join('\\n');"
        "if (!query) return true;"
        "if (matcher) return matcher.test(haystack);"
        "return haystack.toLowerCase().includes(query.toLowerCase());"
        "});"
        "if (!filteredFiles.some((file)=>file.id === currentFile)) {"
        "currentFile = filteredFiles.length ? filteredFiles[0].id : null;"
        "page = 1;"
        "}"
        "renderFileList();"
        "if (!currentFile) {"
        "selectedFileName.textContent = 'No matching parquet file';"
        "selectedFilePath.textContent = query ? `Current filter: ${query}` : 'No parquet files found.';"
        "fileStatus.textContent = '';"
        "schemaWrap.innerHTML = '';"
        "headEl.innerHTML = '';"
        "bodyEl.innerHTML = '';"
        "columnList.innerHTML = '';"
        "currentColumns = [];"
        "columnOrder = [];"
        "return;"
        "}"
        "loadMeta().then(loadPage);"
        "}"
        "async function loadMeta(){"
        "if (!currentFile) return;"
        "const resp = await fetch(`/api/meta?file=${currentFile}`);"
        "const data = await resp.json();"
        "const file = getFileById(currentFile);"
        "totalRows = data.total;"
        "currentColumns = data.columns;"
        "columnOrder = mergeColumnOrder(readSavedColumnOrder(), currentColumns);"
        "schemaWrap.innerHTML = data.schema_html;"
        "renderTableHeader(columnOrder);"
        "renderColumnList();"
        "selectedFileName.textContent = file ? file.name : currentFile;"
        "selectedFilePath.textContent = file ? file.path : '';"
        "fileStatus.textContent = `Rows loaded: ${totalRows}`;"
        "}"
        "async function loadPage(){"
        "if (!currentFile) return;"
        "page = clampPage(page);"
        "const offset = (page - 1) * pageSize;"
        "const params = new URLSearchParams({file: currentFile, offset: String(offset), limit: String(pageSize)});"
        "columnOrder.forEach((column)=>params.append('columns', column));"
        "const resp = await fetch(`/api/page?${params.toString()}`);"
        "const data = await resp.json();"
        "if (data.columns) { columnOrder = data.columns; }"
        "renderTableHeader(columnOrder);"
        "bodyEl.innerHTML = data.rows.join('');"
        "pageStatus.textContent = `Page ${page} / ${totalPages()} (rows ${totalRows})`;"
        "pageInput.value = page;"
        "wireZoom();"
        "}"
        "sizeSelect.value = String(pageSize);"
        "sizeSelect.addEventListener('change', (e)=>{"
        "pageSize = parseInt(e.target.value || '50', 10);"
        "page = 1;"
        "loadPage();"
        "});"
        "document.getElementById('prev-btn').addEventListener('click', ()=>{page -= 1;loadPage();});"
        "document.getElementById('next-btn').addEventListener('click', ()=>{page += 1;loadPage();});"
        "pageInput.addEventListener('change', (e)=>{page = parseInt(e.target.value || '1', 10);loadPage();});"
        "columnReset.addEventListener('click', ()=>{"
        "localStorage.removeItem(COLUMN_ORDER_STORAGE_KEY);"
        "columnOrder = currentColumns.slice();"
        "renderColumnList();"
        "loadPage();"
        "});"
        "const modal = document.getElementById('img-modal');"
        "const modalImg = document.getElementById('modal-img');"
        "const textModal = document.getElementById('text-modal');"
        "const textModalTitle = document.getElementById('text-modal-title');"
        "const textModalContent = document.getElementById('text-modal-content');"
        "function wireZoom(){"
        "document.querySelectorAll('img.zoomable').forEach((img)=>{"
        "img.addEventListener('click', ()=>{"
        "const full = img.dataset.full || img.src;"
        "modalImg.src = full;"
        "modal.style.display = 'flex';"
        "});"
        "});"
        "document.querySelectorAll('.text-trigger').forEach((btn)=>{"
        "btn.addEventListener('click', async ()=>{"
        "textModalTitle.textContent = btn.dataset.title || 'text';"
        "textModalContent.textContent = 'Loading...';"
        "textModal.style.display = 'flex';"
        "if (btn.dataset.rowIndex && btn.dataset.column && currentFile) {"
        "const params = new URLSearchParams({"
        "file: currentFile,"
        "row: btn.dataset.rowIndex,"
        "column: btn.dataset.column,"
        "});"
        "try {"
        "const resp = await fetch(`/api/cell?${params.toString()}`);"
        "const data = await resp.json();"
        "textModalTitle.textContent = data.title || btn.dataset.title || 'text';"
        "textModalContent.textContent = data.content || '';"
        "} catch (error) {"
        "textModalContent.textContent = `Failed to load cell: ${error}`;"
        "}"
        "} else {"
        "textModalContent.textContent = btn.dataset.content || '';"
        "}"
        "});"
        "});"
        "}"
        "modal.addEventListener('click', ()=>{modal.style.display = 'none';});"
        "textModal.addEventListener('click', (e)=>{"
        "if (e.target === textModal) { textModal.style.display = 'none'; }"
        "});"
        "document.getElementById('text-modal-close').addEventListener('click', ()=>{"
        "textModal.style.display = 'none';"
        "});"
        "async function init(){"
        "const resp = await fetch('/api/files');"
        "const data = await resp.json();"
        "allFiles = data.files;"
        "filteredFiles = data.files.slice();"
        "currentFile = filteredFiles.length ? filteredFiles[0].id : null;"
        "renderFileList();"
        "if (!currentFile){fileStatus.textContent = 'No parquet files found.';return;}"
        "await loadMeta();"
        "await loadPage();"
        "fileFilter.addEventListener('input', applyFileFilter);"
        "filterMode.addEventListener('change', applyFileFilter);"
        "}"
        "init();"
        "</script>"
        "</body>"
        "</html>"
    )


class DataStore:
    def __init__(self, files: list[dict[str, str]], columns: Iterable[str] | None):
        self._files = files
        self._columns = list(columns) if columns else None
        self._cache_id: str | None = None
        self._cache_df: pd.DataFrame | None = None

    @property
    def files(self) -> list[dict[str, str]]:
        return [
            {
                "id": f["id"],
                "name": f["name"],
                "basename": f["basename"],
                "path": f["path"],
                "relpath": f["relpath"],
            }
            for f in self._files
        ]

    def _resolve(self, file_id: str) -> str:
        for item in self._files:
            if item["id"] == file_id:
                return item["path"]
        raise KeyError(f"Unknown file id: {file_id}")

    def get_df(self, file_id: str) -> pd.DataFrame:
        if self._cache_id == file_id and self._cache_df is not None:
            return self._cache_df
        path = self._resolve(file_id)
        df = _load_parquet(path, self._columns)
        self._cache_id = file_id
        self._cache_df = df
        return df


def create_app(
    files: list[dict[str, str]],
    cfg: RenderConfig,
    source_hint: str,
    page_size: int,
    columns: Iterable[str] | None,
) -> FastAPI:
    app = FastAPI()
    title = f"Parquet Preview - {source_hint}"
    store = DataStore(files, columns)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _render_index(title, page_size)

    @app.get("/api/files")
    def list_files() -> JSONResponse:
        return JSONResponse({"files": store.files})

    @app.get("/api/meta")
    def meta(file: str = Query(...)) -> JSONResponse:
        df = store.get_df(file)
        schema_html = _render_schema(df)
        columns = [str(col) for col in df.columns]
        return JSONResponse({"total": len(df), "schema_html": schema_html, "columns": columns})

    @app.get("/api/page")
    def page(
        file: str = Query(...),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
        columns: list[str] | None = Query(None),
    ) -> JSONResponse:
        df = store.get_df(file)
        start = offset
        end = min(offset + limit, len(df))
        ordered_columns = [str(col) for col in _resolve_column_order(df, columns)]
        rows = _render_rows(df, start, end, cfg, columns)
        return JSONResponse({"rows": rows, "total": len(df), "columns": ordered_columns})

    @app.get("/api/cell")
    def cell(
        file: str = Query(...),
        row: int = Query(..., ge=0),
        column: str = Query(...),
    ) -> JSONResponse:
        df = store.get_df(file)
        if row >= len(df):
            return JSONResponse(
                {"title": column, "content": f"Row out of range: {row}"},
                status_code=404,
            )

        column_map = {str(col): col for col in df.columns}
        if column not in column_map:
            return JSONResponse(
                {"title": column, "content": f"Column not found: {column}"},
                status_code=404,
            )

        value = df.iloc[row][column_map[column]]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            content = "" if value is None else "NaN"
        else:
            content = _format_full_text(value)
        return JSONResponse({"title": column, "content": content})

    return app


def _collect_paths(parquet_paths: list[str] | None, parquet_dir: str | None) -> list[str]:
    paths: list[str] = []
    if parquet_paths:
        paths.extend(parquet_paths)
    if parquet_dir:
        for name in sorted(os.listdir(parquet_dir)):
            if name.endswith(".parquet"):
                paths.append(os.path.join(parquet_dir, name))
    return paths


def main(args) -> None:
    cfg = RenderConfig(
        max_text_len=args.max_text_len,
        max_collection_len=args.max_collection_len,
        image_thumb_max_side=args.image_thumb_max_side,
        image_full_max_side=args.image_full_max_side,
    )
    parquet_dir = getattr(args, "parquet_dir", None) or getattr(args, "dir", None)
    parquet_paths = _collect_paths(args.parquet_paths, parquet_dir)
    if not parquet_paths:
        raise SystemExit("No parquet files found.")
    files = _build_file_list(parquet_paths)
    source_hint = os.path.basename(parquet_dir) if parquet_dir else "multiple"
    app = create_app(files, cfg, source_hint, args.page_size, args.columns)

    import uvicorn

    host = args.host
    port = args.port
    print(f"Serving preview at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview parquet data (web server)")
    parser.add_argument(
        "parquet_paths",
        nargs="*",
        help="Paths to parquet files",
    )
    parser.add_argument("--dir", help="Directory containing parquet files")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8010, help="Server port")
    parser.add_argument("--columns", nargs="*", help="Subset of columns to load")
    parser.add_argument("--max-text-len", type=int, default=200)
    parser.add_argument("--max-collection-len", type=int, default=2000)
    parser.add_argument("--image-thumb-max-side", type=int, default=256)
    parser.add_argument("--image-full-max-side", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=50, help="Rows per page")
    main(parser.parse_args())
