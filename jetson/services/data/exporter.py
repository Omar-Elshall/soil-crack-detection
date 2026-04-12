"""
exporter.py — Export mission data as GeoJSON or PDF report.
"""

import csv
import io
import json
import os
from datetime import datetime
from typing import Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def mission_to_geojson(mission_dir: str, meta: dict) -> dict:
    """Convert mission detections CSV to GeoJSON FeatureCollection."""
    csv_path = os.path.join(mission_dir, "detections.csv")
    features = []

    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat = float(row.get("lat", 0))
                lon = float(row.get("lon", 0))
                # Skip zero-coordinate rows (no GPS fix)
                if lat == 0.0 and lon == 0.0:
                    continue
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "properties": {
                        "timestamp": row.get("timestamp", ""),
                        "alt_m": float(row.get("alt_m", 0)),
                        "heading_deg": float(row.get("heading_deg", 0)),
                        "crack_ratio_pct": float(row.get("crack_ratio_pct", 0)),
                        "mask_filename": row.get("mask_filename", ""),
                    },
                }
                features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "mission_id": meta.get("id", ""),
            "start_time": meta.get("start_time", ""),
            "end_time": meta.get("end_time", ""),
            "model": meta.get("model", ""),
            "total_detections": meta.get("total_detections", 0),
            "max_coverage_pct": meta.get("max_coverage_pct", 0.0),
            "mean_coverage_pct": meta.get("mean_coverage_pct", 0.0),
        },
        "features": features,
    }
    return geojson


def mission_to_pdf(mission_dir: str, meta: dict) -> Optional[bytes]:
    """Generate a PDF summary report for a mission. Returns bytes or None."""
    if not REPORTLAB_AVAILABLE:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#C4622D"),
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#2C2C2C"),
        spaceBefore=14,
        spaceAfter=4,
    )
    body_style = styles["Normal"]

    story = []

    # Header
    story.append(Paragraph("Soil Crack Detection — Mission Report", title_style))
    story.append(Paragraph(f"Mission ID: {meta.get('id', 'N/A')}", body_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#C4622D")))
    story.append(Spacer(1, 10))

    # Mission summary table
    story.append(Paragraph("Mission Summary", heading_style))

    start = meta.get("start_time", "")
    end = meta.get("end_time", "")
    duration = meta.get("flight_duration_s", 0)
    mins, secs = divmod(int(duration), 60)

    summary_data = [
        ["Field", "Value"],
        ["Status", meta.get("status", "").capitalize()],
        ["Model", meta.get("model", "EfficientCrackNet")],
        ["Start Time", start],
        ["End Time", end if end else "—"],
        ["Flight Duration", f"{mins}m {secs}s"],
        ["Total Detections", str(meta.get("total_detections", 0))],
        ["Max Coverage", f"{meta.get('max_coverage_pct', 0.0):.1f}%"],
        ["Mean Coverage", f"{meta.get('mean_coverage_pct', 0.0):.1f}%"],
    ]

    bbox = meta.get("bbox")
    if bbox:
        summary_data.append(["Bounding Box (lat)", f"{bbox['min_lat']:.6f} → {bbox['max_lat']:.6f}"])
        summary_data.append(["Bounding Box (lon)", f"{bbox['min_lon']:.6f} → {bbox['max_lon']:.6f}"])

    table = Table(summary_data, colWidths=[2.2 * inch, 4.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C4622D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9F6F1"), colors.white]),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    # Detections table
    csv_path = os.path.join(mission_dir, "detections.csv")
    if os.path.exists(csv_path):
        story.append(Paragraph("Detection Log (top 50)", heading_style))
        rows = [["#", "Timestamp", "Lat", "Lon", "Alt (m)", "Coverage %"]]
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 50:
                    break
                rows.append([
                    str(i + 1),
                    row.get("timestamp", "")[:19],
                    f"{float(row.get('lat', 0)):.6f}",
                    f"{float(row.get('lon', 0)):.6f}",
                    row.get("alt_m", "0"),
                    f"{float(row.get('crack_ratio_pct', 0)):.2f}%",
                ])

        det_table = Table(rows, colWidths=[0.4*inch, 2.0*inch, 1.0*inch, 1.0*inch, 0.7*inch, 1.0*inch])
        det_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C2C2C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9F6F1"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(det_table)

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#AAAAAA")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | EfficientCrackNet Soil Crack Detection System",
        ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
