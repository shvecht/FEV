"""Theme definitions for the EDF viewer UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeDefinition:
    """Palette configuration for the viewer.

    Parameters
    ----------
    name:
        Human-friendly display name for the theme.
    pg_background / pg_foreground:
        Colors applied to PyQtGraph backgrounds/foregrounds.
    stylesheet:
        Application stylesheet snippet tailored to this palette.
    curve_colors:
        Sequence of colors used for channel traces.
    stage_curve_color:
        Pen color for the hypnogram/stage trace.
    channel_label_active / channel_label_hidden:
        Inline colors used for per-channel labels.
    preview_colors:
        Optional colors shown in the theme preview swatches.
    """

    name: str
    pg_background: str
    pg_foreground: str
    stylesheet: str
    curve_colors: tuple[str, ...]
    stage_curve_color: str
    channel_label_active: str
    channel_label_hidden: str
    preview_colors: tuple[str, ...] = ()


STYLESHEET_TEMPLATE = """
QMainWindow {{ background-color: {window_bg}; color: {text_primary}; }}
QLabel {{ font-size: 13px; color: {text_primary}; }}
QLabel#absoluteRange, QLabel#windowSummary {{ color: {text_muted}; }}
QLabel#stageSummary {{ color: {stage_summary}; font-weight: 600; }}
QLabel#sourceLabel {{ color: {source_label}; font-style: italic; }}
QFrame#controlPanel {{
    background-color: {control_bg};
    border-right: 1px solid {control_border};
}}
QFrame#telemetryBar {{
    background-color: {telemetry_bg};
    border: 1px solid {telemetry_border};
    border-radius: 8px;
    padding: 8px 12px;
}}
QFrame#telemetryBar QLabel {{ color: {telemetry_text}; }}
QDoubleSpinBox {{
    background-color: {spinbox_bg};
    border: 1px solid {spinbox_border};
    border-radius: 6px;
    padding: 6px 8px;
    color: {spinbox_text};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: transparent;
    border: none;
}}
QComboBox {{
    background-color: {spinbox_bg};
    border: 1px solid {spinbox_border};
    border-radius: 6px;
    padding: 4px 8px;
    color: {spinbox_text};
}}
QComboBox::drop-down {{ border: none; }}
QLineEdit {{
    background-color: {spinbox_bg};
    border: 1px solid {spinbox_border};
    border-radius: 6px;
    padding: 6px 8px;
    color: {spinbox_text};
}}
QLineEdit::placeholder {{
    color: {text_muted};
}}
QPushButton#fileSelectButton {{
    background-color: {file_button_bg};
    border: 1px solid {file_button_border};
    border-radius: 6px;
    padding: 8px 12px;
    color: {file_button_text};
    font-weight: 600;
}}
QPushButton#fileSelectButton:hover {{
    background-color: {file_button_bg_hover};
    border-color: {file_button_border_hover};
}}
QPushButton#fileSelectButton:pressed {{
    background-color: {file_button_bg_pressed};
}}
QGroupBox#primaryControls,
QGroupBox#annotationSection {{
    background-color: {groupbox_bg};
    border: 1px solid {groupbox_border};
    border-radius: 10px;
    margin-top: 12px;
    padding: 18px 16px 14px 16px;
}}
QGroupBox#primaryControls::title,
QGroupBox#annotationSection::title {{
    subcontrol-origin: margin;
    left: 18px;
    padding: 0 6px;
    color: {groupbox_title};
    font-weight: 600;
}}
QGroupBox#primaryControls PushButton,
QGroupBox#annotationSection PushButton {{
    background-color: {group_button_bg};
    border: 1px solid {group_button_border};
    border-radius: 6px;
    padding: 6px 10px;
    color: {group_button_text};
}}
QGroupBox#primaryControls PushButton:hover,
QGroupBox#annotationSection PushButton:hover {{
    background-color: {group_button_bg_hover};
}}
QGroupBox#primaryControls PushButton:pressed,
QGroupBox#annotationSection PushButton:pressed {{
    background-color: {group_button_bg_pressed};
}}
QFrame#channelSection {{
    background-color: {channel_section_bg};
    border: 1px solid {channel_section_border};
    border-radius: 10px;
    margin-top: 12px;
    padding: 16px 16px 12px 16px;
}}
QFrame#channelSection QCheckBox {{
    color: {channel_checkbox};
}}
QGroupBox#annotationSection QCheckBox {{
    color: {text_primary};
}}
QGroupBox#annotationSection QListWidget {{
    background-color: {annotation_list_bg};
    border: 1px solid {annotation_list_border};
    border-radius: 6px;
    color: {text_primary};
}}
QGroupBox#annotationSection QListWidget::item:selected {{
    background-color: {file_button_bg};
    color: {file_button_text};
}}
QGroupBox#annotationSection QListWidget::item:hover {{
    background-color: {tool_button_bg_hover};
}}
QFrame#prefetchSection,
QFrame#appearanceSection {{
    background-color: {prefetch_bg};
    border: 1px solid {prefetch_border};
    border-radius: 10px;
    margin-top: 12px;
    padding: 16px 16px 12px 16px;
}}
QFrame#channelSection QToolButton#collapsibleSectionToggle,
QFrame#prefetchSection QToolButton#collapsibleSectionToggle,
QFrame#appearanceSection QToolButton#collapsibleSectionToggle {{
    background-color: transparent;
    border: none;
    border-radius: 0px;
    padding: 0px;
    color: {prefetch_toggle_text};
    font-weight: 600;
    text-align: left;
}}
QFrame#channelSection QToolButton#collapsibleSectionToggle:hover,
QFrame#prefetchSection QToolButton#collapsibleSectionToggle:hover,
QFrame#appearanceSection QToolButton#collapsibleSectionToggle:hover {{
    color: {prefetch_toggle_hover};
    background-color: transparent;
}}
QToolButton {{
    background-color: {tool_button_bg};
    border: 1px solid {tool_button_border};
    border-radius: 4px;
    padding: 4px 8px;
    color: {tool_button_text};
}}
QToolButton:hover {{
    background-color: {tool_button_bg_hover};
}}
QToolButton:pressed {{
    background-color: {tool_button_bg_pressed};
}}
QProgressBar#ingestBar {{
    background-color: {ingest_bar_bg};
    border: 1px solid {ingest_bar_border};
    border-radius: 6px;
    padding: 3px;
    color: {text_primary};
}}
QProgressBar#ingestBar::chunk {{
    background-color: {ingest_bar_chunk};
    border-radius: 4px;
}}
QScrollArea {{ background-color: {scroll_bg}; }}
"""


def _make_stylesheet(palette: dict[str, str]) -> str:
    return STYLESHEET_TEMPLATE.format(**palette)


DEFAULT_THEME = "Midnight"


THEMES: dict[str, ThemeDefinition] = {
    "Midnight": ThemeDefinition(
        name="Midnight",
        pg_background="#0b111c",
        pg_foreground="#e3e7f3",
        stylesheet=_make_stylesheet(
            {
                "window_bg": "#0b111c",
                "text_primary": "#e6ebf5",
                "text_muted": "#9ba9bf",
                "stage_summary": "#d2def6",
                "source_label": "#9ba9bf",
                "control_bg": "#131b2b",
                "control_border": "#1f2a3d",
                "telemetry_bg": "#121a24",
                "telemetry_border": "#1f2a3d",
                "telemetry_text": "#c5d1ec",
                "spinbox_bg": "#121a2a",
                "spinbox_border": "#1f2a3d",
                "spinbox_text": "#f0f4ff",
                "file_button_bg": "#1a2436",
                "file_button_border": "#27324a",
                "file_button_text": "#f3f6ff",
                "file_button_bg_hover": "#22304a",
                "file_button_border_hover": "#39507a",
                "file_button_bg_pressed": "#182235",
                "groupbox_bg": "#141e30",
                "groupbox_border": "#24324b",
                "groupbox_title": "#a7b4cf",
                "group_button_bg": "#1c273a",
                "group_button_border": "#2b3850",
                "group_button_text": "#e1e9ff",
                "group_button_bg_hover": "#263755",
                "group_button_bg_pressed": "#142033",
                "channel_section_bg": "#141e30",
                "channel_section_border": "#24324b",
                "channel_checkbox": "#dfe7ff",
                "annotation_list_bg": "#0d1420",
                "annotation_list_border": "#1f2a3d",
                "prefetch_bg": "#141e30",
                "prefetch_border": "#24324b",
                "tool_button_bg": "#1a2333",
                "tool_button_border": "#263247",
                "tool_button_text": "#dfe7ff",
                "tool_button_bg_hover": "#25314a",
                "tool_button_bg_pressed": "#172132",
                "prefetch_toggle_text": "#a7b4cf",
                "prefetch_toggle_hover": "#d0dcf5",
                "ingest_bar_bg": "#121a24",
                "ingest_bar_border": "#1f2a3d",
                "ingest_bar_chunk": "#3d6dff",
                "scroll_bg": "#0b111c",
            }
        ),
        curve_colors=(
            "#5f8bff",
            "#f4b860",
            "#5dd39e",
            "#f57f7f",
            "#c792ea",
            "#3fb8ff",
        ),
        stage_curve_color="#5f8bff",
        channel_label_active="#dfe7ff",
        channel_label_hidden="#6c788f",
        preview_colors=("#5f8bff", "#f4b860", "#5dd39e"),
    ),
    "Dawn": ThemeDefinition(
        name="Dawn",
        pg_background="#F5F7FA",
        pg_foreground="#0F172A",
        stylesheet=_make_stylesheet(
            {
                # Base canvas & text
                "window_bg": "#F5F7FA",
                "text_primary": "#0F172A",
                "text_muted": "#5B6573",
                "stage_summary": "#1E3A8A",
                "source_label": "#64748B",

                # Left control rail
                "control_bg": "#FFFFFF",
                "control_border": "#D9E1EA",

                # Telemetry pill
                "telemetry_bg": "#EEF2F7",
                "telemetry_border": "#D9E1EA",
                "telemetry_text": "#0F172A",

                # Inputs (spinbox/combos/line-edits share these)
                "spinbox_bg": "#FFFFFF",
                "spinbox_border": "#CBD5E1",
                "spinbox_text": "#0F172A",

                # Primary action button (file select)
                "file_button_bg": "#E7EEF9",
                "file_button_border": "#C5D1EA",
                "file_button_text": "#0F172A",
                "file_button_bg_hover": "#D9E4F7",
                "file_button_border_hover": "#AFC1E6",
                "file_button_bg_pressed": "#CAD7F1",

                # Group boxes
                "groupbox_bg": "#FFFFFF",
                "groupbox_border": "#D9E1EA",
                "groupbox_title": "#334155",
                "group_button_bg": "#F0F4FA",
                "group_button_border": "#D4DEED",
                "group_button_text": "#1F2A44",
                "group_button_bg_hover": "#E6ECF7",
                "group_button_bg_pressed": "#D9E4F3",

                # Channel list
                "channel_section_bg": "#FFFFFF",
                "channel_section_border": "#D9E1EA",
                "channel_checkbox": "#0F172A",

                # Annotations list (slight tint so it stands apart from white)
                "annotation_list_bg": "#F9FBFD",
                "annotation_list_border": "#D9E1EA",

                # Prefetch & Appearance frames
                "prefetch_bg": "#FFFFFF",
                "prefetch_border": "#D9E1EA",

                # Tool buttons (small icon buttons)
                "tool_button_bg": "#E9EEF8",
                "tool_button_border": "#CBD5E1",
                "tool_button_text": "#0F172A",
                "tool_button_bg_hover": "#DFE7F7",
                "tool_button_bg_pressed": "#CFDBF1",

                # Expanders
                "prefetch_toggle_text": "#334155",
                "prefetch_toggle_hover": "#1F2A44",

                # Progress bar (ingest)
                "ingest_bar_bg": "#EEF2F7",
                "ingest_bar_border": "#CBD5E1",
                "ingest_bar_chunk": "#3B82F6",

                # Scroll areas
                "scroll_bg": "#F2F4F8",
            }
        ),
        # Curves: readable on a light canvas
        curve_colors=(
            "#1D4ED8",  # Blue
            "#B45309",  # Amber (darker for light bg)
            "#059669",  # Green
            "#9333EA",  # Violet
            "#0EA5E9",  # Sky
            "#DC2626",  # Red (alerts)
        ),
        stage_curve_color="#1D4ED8",
        channel_label_active="#0F172A",
        channel_label_hidden="#6B7280",
        preview_colors=("#1D4ED8", "#B45309", "#059669"),
    ),
    "Slate": ThemeDefinition(
        name="Slate",
        pg_background="#121417",
        pg_foreground="#e4e7eb",
        stylesheet=_make_stylesheet(
            {
                "window_bg": "#121417",
                "text_primary": "#e4e7eb",
                "text_muted": "#a0a6b0",
                "stage_summary": "#f2f5ff",
                "source_label": "#9aa1ad",
                "control_bg": "#181b20",
                "control_border": "#272b33",
                "telemetry_bg": "#1b1f24",
                "telemetry_border": "#2c313a",
                "telemetry_text": "#d0d5dd",
                "spinbox_bg": "#1a1f26",
                "spinbox_border": "#2b303a",
                "spinbox_text": "#f1f3f6",
                "file_button_bg": "#21262f",
                "file_button_border": "#313844",
                "file_button_text": "#f3f5f8",
                "file_button_bg_hover": "#2a303b",
                "file_button_border_hover": "#3d4554",
                "file_button_bg_pressed": "#1d222a",
                "groupbox_bg": "#192027",
                "groupbox_border": "#282e38",
                "groupbox_title": "#b2b8c2",
                "group_button_bg": "#232934",
                "group_button_border": "#343b47",
                "group_button_text": "#e6e9ef",
                "group_button_bg_hover": "#2e3542",
                "group_button_bg_pressed": "#202631",
                "channel_section_bg": "#181c22",
                "channel_section_border": "#282e38",
                "channel_checkbox": "#f0f2f6",
                "annotation_list_bg": "#131821",
                "annotation_list_border": "#242933",
                "prefetch_bg": "#181c22",
                "prefetch_border": "#282e38",
                "tool_button_bg": "#222831",
                "tool_button_border": "#303743",
                "tool_button_text": "#e4e7eb",
                "tool_button_bg_hover": "#2d3441",
                "tool_button_bg_pressed": "#1b1f27",
                "prefetch_toggle_text": "#b2b8c2",
                "prefetch_toggle_hover": "#e4e7eb",
                "ingest_bar_bg": "#1b1f24",
                "ingest_bar_border": "#2c313a",
                "ingest_bar_chunk": "#6b8cff",
                "scroll_bg": "#121417",
            }
        ),
        curve_colors=(
            "#4f7cff",
            "#f0b35b",
            "#4fbf9f",
            "#f26d6d",
            "#c38bff",
            "#46a7ff",
        ),
        stage_curve_color="#4f7cff",
        channel_label_active="#f0f2f6",
        channel_label_hidden="#707884",
        preview_colors=("#4f7cff", "#f0b35b", "#4fbf9f"),
    ),

    "OliveGrove": ThemeDefinition(
        name="OliveGrove",
        pg_background="#FEFAE0",
        pg_foreground="#283618",
        stylesheet=_make_stylesheet(
            {
                "window_bg": "#FEFAE0",
                "text_primary": "#283618",
                "text_muted": "#606C38",
                "stage_summary": "#606C38",
                "source_label": "#BC6C25",
                "control_bg": "#FFFFFF",
                "control_border": "#E6E2C8",
                "telemetry_bg": "#FFFDF0",
                "telemetry_border": "#E6E2C8",
                "telemetry_text": "#283618",
                "spinbox_bg": "#FFFFFF",
                "spinbox_border": "#E0DCC5",
                "spinbox_text": "#283618",
                "file_button_bg": "#FFF2E0",
                "file_button_border": "#F1DEBE",
                "file_button_text": "#283618",
                "file_button_bg_hover": "#FFE8C8",
                "file_button_border_hover": "#E9D0A8",
                "file_button_bg_pressed": "#FADDAF",
                "groupbox_bg": "#FFFFFF",
                "groupbox_border": "#E6E2C8",
                "groupbox_title": "#283618",
                "group_button_bg": "#F7F3E0",
                "group_button_border": "#E6E2C8",
                "group_button_text": "#283618",
                "group_button_bg_hover": "#EFE8D0",
                "group_button_bg_pressed": "#E7E0C2",
                "channel_section_bg": "#FFFFFF",
                "channel_section_border": "#E6E2C8",
                "channel_checkbox": "#283618",
                "annotation_list_bg": "#FFF3DA",
                "annotation_list_border": "#E6E2C8",
                "prefetch_bg": "#FFFFFF",
                "prefetch_border": "#E6E2C8",
                "tool_button_bg": "#F6E8D6",
                "tool_button_border": "#E9D7BD",
                "tool_button_text": "#283618",
                "tool_button_bg_hover": "#F0DCBE",
                "tool_button_bg_pressed": "#E9D0A8",
                "prefetch_toggle_text": "#606C38",
                "prefetch_toggle_hover": "#283618",
                "ingest_bar_bg": "#FFFDF0",
                "ingest_bar_border": "#E6E2C8",
                "ingest_bar_chunk": "#BC6C25",
                "scroll_bg": "#FEFAE0",
            }
        ),
        curve_colors=(
            "#606C38",  # olive
            "#DDA15E",  # sand
            "#BC6C25",  # rust
            "#283618",  # deep olive
            "#8DAA5B",  # derived accent
            "#9C6F34",  # derived accent
        ),
        stage_curve_color="#606C38",
        channel_label_active="#283618",
        channel_label_hidden="#787D66",
        preview_colors=("#606C38", "#DDA15E", "#BC6C25"),
    ),

    "Harvest": ThemeDefinition(
        name="Harvest",
        pg_background="#FEFAE0",
        pg_foreground="#3A3A2A",
        stylesheet=_make_stylesheet(
            {
                "window_bg": "#FEFAE0",
                "text_primary": "#3A3A2A",
                "text_muted": "#6B6B5A",
                "stage_summary": "#D4A373",
                "source_label": "#A78C5A",
                "control_bg": "#FFFFFF",
                "control_border": "#E7E3CE",
                "telemetry_bg": "#FDF7E7",
                "telemetry_border": "#E7E3CE",
                "telemetry_text": "#3A3A2A",
                "spinbox_bg": "#FFFFFF",
                "spinbox_border": "#E1DEC9",
                "spinbox_text": "#3A3A2A",
                "file_button_bg": "#FAEDCD",
                "file_button_border": "#EAD9A7",
                "file_button_text": "#3A3A2A",
                "file_button_bg_hover": "#F2E1B4",
                "file_button_border_hover": "#E1CB8D",
                "file_button_bg_pressed": "#E7D39C",
                "groupbox_bg": "#FFFFFF",
                "groupbox_border": "#E7E3CE",
                "groupbox_title": "#4B4B3A",
                "group_button_bg": "#E9EDC9",
                "group_button_border": "#D9DEBB",
                "group_button_text": "#3A3A2A",
                "group_button_bg_hover": "#DFE5B4",
                "group_button_bg_pressed": "#D4DAA2",
                "channel_section_bg": "#FFFFFF",
                "channel_section_border": "#E7E3CE",
                "channel_checkbox": "#3A3A2A",
                "annotation_list_bg": "#FFF7E8",
                "annotation_list_border": "#E7E3CE",
                "prefetch_bg": "#FFFFFF",
                "prefetch_border": "#E7E3CE",
                "tool_button_bg": "#F3E7D4",
                "tool_button_border": "#E6D6BE",
                "tool_button_text": "#3A3A2A",
                "tool_button_bg_hover": "#EAD9C0",
                "tool_button_bg_pressed": "#E0CBAA",
                "prefetch_toggle_text": "#6B6B5A",
                "prefetch_toggle_hover": "#3A3A2A",
                "ingest_bar_bg": "#FDF7E7",
                "ingest_bar_border": "#E7E3CE",
                "ingest_bar_chunk": "#D4A373",
                "scroll_bg": "#FEFAE0",
            }
        ),
        curve_colors=(
            "#CCD5AE",
            "#E9EDC9",
            "#FAEDCD",
            "#D4A373",
            "#A98467",  # derived
            "#6B705C",  # derived
        ),
        stage_curve_color="#D4A373",
        channel_label_active="#3A3A2A",
        channel_label_hidden="#6B6B5A",
        preview_colors=("#CCD5AE", "#FAEDCD", "#D4A373"),
    ),

    "Glacier": ThemeDefinition(
        name="Glacier",
        pg_background="#023047",
        pg_foreground="#E6F4FF",
        stylesheet=_make_stylesheet(
            {
                "window_bg": "#061C2A",
                "text_primary": "#E6F4FF",
                "text_muted": "#99BBD1",
                "stage_summary": "#8ECAE6",
                "source_label": "#219EBC",
                "control_bg": "#0B2636",
                "control_border": "#133447",
                "telemetry_bg": "#0A2231",
                "telemetry_border": "#133447",
                "telemetry_text": "#CDE7F6",
                "spinbox_bg": "#0B2636",
                "spinbox_border": "#18475F",
                "spinbox_text": "#E6F4FF",
                "file_button_bg": "#133E55",
                "file_button_border": "#1B5671",
                "file_button_text": "#E6F4FF",
                "file_button_bg_hover": "#1B5671",
                "file_button_border_hover": "#2B6F8C",
                "file_button_bg_pressed": "#10405A",
                "groupbox_bg": "#0C2534",
                "groupbox_border": "#18475F",
                "groupbox_title": "#BFE6FA",
                "group_button_bg": "#0F2C3D",
                "group_button_border": "#1B4D65",
                "group_button_text": "#E6F4FF",
                "group_button_bg_hover": "#17465F",
                "group_button_bg_pressed": "#0E2A3A",
                "channel_section_bg": "#0C2534",
                "channel_section_border": "#18475F",
                "channel_checkbox": "#E6F4FF",
                "annotation_list_bg": "#0A1E2C",
                "annotation_list_border": "#1A3A4D",
                "prefetch_bg": "#0C2534",
                "prefetch_border": "#18475F",
                "tool_button_bg": "#0F2C3D",
                "tool_button_border": "#1B4D65",
                "tool_button_text": "#E6F4FF",
                "tool_button_bg_hover": "#17465F",
                "tool_button_bg_pressed": "#0E2A3A",
                "prefetch_toggle_text": "#BFE6FA",
                "prefetch_toggle_hover": "#E6F4FF",
                "ingest_bar_bg": "#0A2231",
                "ingest_bar_border": "#123245",
                "ingest_bar_chunk": "#FFB703",
                "scroll_bg": "#061C2A",
            }
        ),
        curve_colors=(
            "#8ECAE6",
            "#219EBC",
            "#FFB703",
            "#FB8500",
            "#E6E6E6",
            "#A3D2E3",
        ),
        stage_curve_color="#8ECAE6",
        channel_label_active="#E6F4FF",
        channel_label_hidden="#99BBD1",
        preview_colors=("#8ECAE6", "#219EBC", "#FFB703"),
    ),

    "AquaRamp": ThemeDefinition(
        name="AquaRamp",
        pg_background="#D9ED92",
        pg_foreground="#184E77",
        stylesheet=_make_stylesheet(
            {
                "window_bg": "#EEF8D1",
                "text_primary": "#184E77",
                "text_muted": "#2F6C90",
                "stage_summary": "#168AAD",
                "source_label": "#1A759F",
                "control_bg": "#FFFFFF",
                "control_border": "#CFEAC8",
                "telemetry_bg": "#F5FAE6",
                "telemetry_border": "#CFEAC8",
                "telemetry_text": "#184E77",
                "spinbox_bg": "#FFFFFF",
                "spinbox_border": "#C9E5C2",
                "spinbox_text": "#184E77",
                "file_button_bg": "#E0F5E8",
                "file_button_border": "#BFE9DC",
                "file_button_text": "#184E77",
                "file_button_bg_hover": "#D1EFE0",
                "file_button_border_hover": "#A8E0D1",
                "file_button_bg_pressed": "#BCE3C8",
                "groupbox_bg": "#FFFFFF",
                "groupbox_border": "#CFEAC8",
                "groupbox_title": "#1E6091",
                "group_button_bg": "#E8F7EE",
                "group_button_border": "#C7EEDF",
                "group_button_text": "#184E77",
                "group_button_bg_hover": "#D7F0E6",
                "group_button_bg_pressed": "#C5E7DC",
                "channel_section_bg": "#FFFFFF",
                "channel_section_border": "#CFEAC8",
                "channel_checkbox": "#184E77",
                "annotation_list_bg": "#F3FAF5",
                "annotation_list_border": "#CFEAC8",
                "prefetch_bg": "#FFFFFF",
                "prefetch_border": "#CFEAC8",
                "tool_button_bg": "#E2F4F6",
                "tool_button_border": "#C5E8ED",
                "tool_button_text": "#184E77",
                "tool_button_bg_hover": "#D4EDF0",
                "tool_button_bg_pressed": "#C3E3E7",
                "prefetch_toggle_text": "#1A759F",
                "prefetch_toggle_hover": "#184E77",
                "ingest_bar_bg": "#F5FAE6",
                "ingest_bar_border": "#CFEAC8",
                "ingest_bar_chunk": "#168AAD",
                "scroll_bg": "#EEF8D1",
            }
        ),
        curve_colors=(
            "#99D98C",
            "#76C893",
            "#52B69A",
            "#34A0A4",
            "#1A759F",
            "#184E77",
        ),
        stage_curve_color="#168AAD",
        channel_label_active="#184E77",
        channel_label_hidden="#2F6C90",
        preview_colors=("#99D98C", "#34A0A4", "#1A759F"),
    ),
    "CarbonNight": ThemeDefinition(
        name="CarbonNight",
        pg_background="#1b2129",
        pg_foreground="#e6edf5",
        stylesheet=_make_stylesheet(
            {
                "window_bg": "#161a1f",
                "text_primary": "#e6edf5",
                "text_muted": "#9aa3ad",
                "stage_summary": "#8bd4c6",
                "source_label": "#9aa3ad",
                "control_bg": "#1d2229",
                "control_border": "#2a313b",
                "telemetry_bg": "#151a1f",
                "telemetry_border": "#2a313b",
                "telemetry_text": "#d2d8e0",
                "spinbox_bg": "#1c222a",
                "spinbox_border": "#2b3340",
                "spinbox_text": "#f2f6ff",
                "file_button_bg": "#223041",
                "file_button_border": "#2f3e52",
                "file_button_text": "#eaf2ff",
                "file_button_bg_hover": "#2c3d52",
                "file_button_border_hover": "#3d5373",
                "file_button_bg_pressed": "#1e2a3a",
                "groupbox_bg": "#1b2027",
                "groupbox_border": "#2a303a",
                "groupbox_title": "#b6c0cc",
                "group_button_bg": "#232a34",
                "group_button_border": "#313a46",
                "group_button_text": "#e7edf6",
                "group_button_bg_hover": "#2c3542",
                "group_button_bg_pressed": "#1f2630",
                "channel_section_bg": "#1b2027",
                "channel_section_border": "#2a303a",
                "channel_checkbox": "#f1f5fb",
                "annotation_list_bg": "#14181f",
                "annotation_list_border": "#242a34",
                "prefetch_bg": "#1b2027",
                "prefetch_border": "#2a303a",
                "tool_button_bg": "#202732",
                "tool_button_border": "#2e3744",
                "tool_button_text": "#e6edf5",
                "tool_button_bg_hover": "#2a3340",
                "tool_button_bg_pressed": "#1a2029",
                "prefetch_toggle_text": "#b6c0cc",
                "prefetch_toggle_hover": "#e6edf5",
                "ingest_bar_bg": "#151a1f",
                "ingest_bar_border": "#242b36",
                "ingest_bar_chunk": "#4f7cff",
                "scroll_bg": "#161a1f",
            }
        ),
        curve_colors=(
            "#31c7b2",  # teal
            "#f0b429",  # amber
            "#60a5fa",  # blue
            "#a78bfa",  # violet
            "#22d3ee",  # cyan
            "#f472b6",  # pink
        ),
        stage_curve_color="#60a5fa",
        channel_label_active="#e6edf5",
        channel_label_hidden="#718096",
        preview_colors=("#31c7b2", "#f0b429", "#60a5fa"),
    ),
}
