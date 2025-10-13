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
QGroupBox#annotationSection QListWidget {{
    background-color: {annotation_list_bg};
    border: 1px solid {annotation_list_border};
    border-radius: 6px;
}}
QFrame#prefetchSection,
QFrame#appearanceSection {{
    background-color: {prefetch_bg};
    border: 1px solid {prefetch_border};
    border-radius: 10px;
    margin-top: 12px;
    padding: 16px 16px 12px 16px;
}}
QFrame#prefetchSection QToolButton,
QFrame#appearanceSection QToolButton {{
    background-color: transparent;
    border: none;
    border-radius: 0px;
    padding: 0px;
    color: {prefetch_toggle_text};
    font-weight: 600;
}}
QFrame#prefetchSection QToolButton:hover,
QFrame#appearanceSection QToolButton:hover {{
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
                "annotation_list_bg": "#0f1724",
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
        pg_background="#f7f7fb",
        pg_foreground="#1a2433",
        stylesheet=_make_stylesheet(
            {
                "window_bg": "#f7f7fb",
                "text_primary": "#1a2433",
                "text_muted": "#5c6475",
                "stage_summary": "#2b4778",
                "source_label": "#667089",
                "control_bg": "#ffffff",
                "control_border": "#d6d9e3",
                "telemetry_bg": "#f0f2f8",
                "telemetry_border": "#d6d9e3",
                "telemetry_text": "#354157",
                "spinbox_bg": "#ffffff",
                "spinbox_border": "#c7ccda",
                "spinbox_text": "#1a2433",
                "file_button_bg": "#e7ecfa",
                "file_button_border": "#c1c8e4",
                "file_button_text": "#1a2433",
                "file_button_bg_hover": "#dbe3f8",
                "file_button_border_hover": "#a9b6e0",
                "file_button_bg_pressed": "#ccd6f3",
                "groupbox_bg": "#ffffff",
                "groupbox_border": "#d6d9e3",
                "groupbox_title": "#46557a",
                "group_button_bg": "#edf1fb",
                "group_button_border": "#ccd3eb",
                "group_button_text": "#24324f",
                "group_button_bg_hover": "#e0e7fa",
                "group_button_bg_pressed": "#d0d9f3",
                "channel_section_bg": "#ffffff",
                "channel_section_border": "#d6d9e3",
                "channel_checkbox": "#1a2433",
                "annotation_list_bg": "#f3f5fb",
                "annotation_list_border": "#d1d6e6",
                "prefetch_bg": "#ffffff",
                "prefetch_border": "#d6d9e3",
                "tool_button_bg": "#e6ebf9",
                "tool_button_border": "#cbd2e7",
                "tool_button_text": "#1f2a3d",
                "tool_button_bg_hover": "#d8e0f5",
                "tool_button_bg_pressed": "#c8d2ed",
                "prefetch_toggle_text": "#46557a",
                "prefetch_toggle_hover": "#2f3f62",
                "ingest_bar_bg": "#f0f2f8",
                "ingest_bar_border": "#cbd0de",
                "ingest_bar_chunk": "#4f6de0",
                "scroll_bg": "#f7f7fb",
            }
        ),
        curve_colors=(
            "#3461c1",
            "#cf6b2b",
            "#1d936c",
            "#d94f67",
            "#8758c7",
            "#2d7bd3",
        ),
        stage_curve_color="#3461c1",
        channel_label_active="#24324f",
        channel_label_hidden="#80889a",
        preview_colors=("#3461c1", "#cf6b2b", "#1d936c"),
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
                "groupbox_bg": "#181c22",
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
                "annotation_list_bg": "#14171d",
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
}

