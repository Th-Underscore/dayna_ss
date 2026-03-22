import gradio as gr
import extensions.dayna_ss.shared as dss_shared
from extensions.dayna_ss.ui import utils
from extensions.dayna_ss.utils.helpers import load_json
from pathlib import Path

from extensions.dayna_ss.utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _WARNING,
    _RESET,
    _DEBUG,
)


def _refresh_template_settings():
    """Re-initialize template settings from format_templates.json."""
    try:
        dss_dir = Path(__file__).parent.parent
        template_path = dss_dir / "user_data" / "example" / "format_templates.json"
        templates = load_json(template_path) or {}
        
        for key in templates:
            setting_key = f"template_{key}"
            if setting_key not in dss_shared.settings:
                dss_shared.settings[setting_key] = ""
        
        print(f"{_DEBUG}Refreshed template settings: {len(templates)} templates{_RESET}")
        return list(dss_shared.settings.keys())
    except Exception as e:
        print(f"{_WARNING}Failed to refresh template settings: {e}{_RESET}")
        return []


def create_ui():
    """Create UI for custom Jinja templates."""
    import re
    
    template_keys = [k for k in dss_shared.settings.keys() if k.startswith("template_")]
    template_data = [(k.replace("template_", ""), k) for k in template_keys]
    template_data.sort()
    
    with gr.Tab("Templates", elem_id="dss-templates"):
        gr.Markdown("## Custom Jinja Templates")
        gr.Markdown("Override default format templates for different data types. Leave empty to use defaults.")
        
        cols = 2
        for i in range(0, len(template_data), cols):
            with gr.Row():
                for j in range(cols):
                    if i + j < len(template_data):
                        data_type, key = template_data[i + j]
                        label = re.sub(r'(?<!^)(?=[A-Z])', ' ', data_type).title()
                        dss_shared.gradio[key] = gr.Textbox(
                            label=f"{label} Template",
                            placeholder=f"Jinja template for {data_type}...",
                            lines=4,
                            interactive=True,
                        )
        
        with gr.Row():
            dss_shared.gradio["refresh_templates_btn"] = gr.Button("🔄 Refresh", variant="secondary")
            dss_shared.gradio["load_templates_btn"] = gr.Button("💾 Save to Preset", variant="secondary")
            dss_shared.gradio["reset_templates_btn"] = gr.Button("🔁 Reset to Defaults", variant="stop")
        
        gr.Markdown("### Template Variables")
        gr.Markdown("""
        Available variables in templates:
        - `{{data}}` - The raw data dict/list
        - `{{path}}` - The path prefix for markers
        - `{{defaults}}` - Schema default values (when available)
        
        For lists: use `{% for item in data %}...{% endfor %}`
        For dicts: use `{% for key, value in data.items() %}...{% endfor %}`
        
        Include path markers manually where needed using `<<<<<<<<<<<< {{path}}.fieldname`
        """)
        
        gr.Markdown("### Current Template Keys")
        gr.Markdown(f"Loaded: {', '.join([k for k in dss_shared.settings.keys() if k.startswith('template_')])}")


def create_event_handlers():
    """Create event handlers for template UI."""
    def on_refresh():
        _refresh_template_settings()
        return "Templates refreshed!"
    
    dss_shared.gradio["refresh_templates_btn"].click(
        on_refresh,
        outputs=dss_shared.gradio.get("refresh_templates_btn") if "refresh_templates_btn" in dss_shared.gradio else None,
    )