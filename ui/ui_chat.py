import json
from pathlib import Path

import gradio as gr
from PIL import Image

from modules import chat, shared, ui

import extensions.dayna_ss.shared as dss_shared
from extensions.dayna_ss.ui import utils
from extensions.dayna_ss.ui.utils import gradio


def create_ui():
    mu = shared.args.multi_user

    dss_shared.gradio["unique_id"] = gr.Radio(
        label="",
        elem_classes=["slim-dropdown", "pretty_scrollbar"],
        interactive=not mu,
        elem_id="dss-past-chats",
        visible=False,
    )


def create_chat_settings_ui():
    mu = shared.args.multi_user

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Tab("Character"):
                    with gr.Row():
                        available_characters = utils.get_available_characters()
                        dss_shared.gradio["character_menu"] = gr.Dropdown(
                            value=None,
                            choices=available_characters,
                            label="Character",
                            elem_id="dss-character-menu",
                            info="Used in chat and chat-instruct modes.",
                            elem_classes="slim-dropdown",
                        )
                        ui.create_refresh_button(
                            dss_shared.gradio["character_menu"],
                            lambda: None,
                            lambda: {"choices": available_characters},
                            "refresh-button",
                            interactive=not mu,
                        )
                        dss_shared.gradio["save_character"] = gr.Button(
                            "💾",
                            elem_classes="refresh-button",
                            elem_id="save-character",
                            interactive=not mu,
                        )
                        dss_shared.gradio["delete_character"] = gr.Button(
                            "🗑️", elem_classes="refresh-button", interactive=not mu
                        )

                    dss_shared.gradio["name2"] = gr.Textbox(value="", lines=1, label="Character's name")
                    dss_shared.gradio["context"] = gr.Textbox(
                        value="", lines=10, label="Context", elem_classes=["add_scrollbar"]
                    )
                    dss_shared.gradio["greeting"] = gr.Textbox(
                        value="", lines=5, label="Greeting", elem_classes=["add_scrollbar"]
                    )

                with gr.Tab("User"):
                    dss_shared.gradio["name1"] = gr.Textbox(value=dss_shared.settings["name1"], lines=1, label="Name")
                    dss_shared.gradio["user_bio"] = gr.Textbox(
                        value=dss_shared.settings["user_bio"],
                        lines=10,
                        label="Description",
                        info="Here you can optionally write a description of yourself.",
                        placeholder="{{user}}'s personality: ...",
                        elem_classes=["add_scrollbar"],
                    )

                with gr.Tab("Upload character"):
                    with gr.Tab("YAML or JSON"):
                        with gr.Row():
                            dss_shared.gradio["upload_json"] = gr.File(
                                type="binary",
                                file_types=[".json", ".yaml"],
                                label="JSON or YAML File",
                                interactive=not mu,
                            )
                            dss_shared.gradio["upload_img_bot"] = gr.Image(
                                type="pil", label="Profile Picture (optional)", interactive=not mu
                            )

                        dss_shared.gradio["Submit character"] = gr.Button(value="Submit", interactive=False)

                    with gr.Tab("TavernAI PNG"):
                        with gr.Row():
                            with gr.Column():
                                dss_shared.gradio["upload_img_tavern"] = gr.Image(
                                    type="pil",
                                    label="TavernAI PNG File",
                                    elem_id="upload_img_tavern",
                                    interactive=not mu,
                                )
                                dss_shared.gradio["tavern_json"] = gr.State()
                            with gr.Column():
                                dss_shared.gradio["tavern_name"] = gr.Textbox(
                                    value="", lines=1, label="Name", interactive=False
                                )
                                dss_shared.gradio["tavern_desc"] = gr.Textbox(
                                    value="",
                                    lines=10,
                                    label="Description",
                                    interactive=False,
                                    elem_classes=["add_scrollbar"],
                                )

                        dss_shared.gradio["Submit tavern character"] = gr.Button(value="Submit", interactive=False)

            with gr.Column(scale=1):
                dss_shared.gradio["character_picture"] = gr.Image(label="Character picture", type="pil", interactive=not mu)
                dss_shared.gradio["your_picture"] = gr.Image(
                    label="Your picture",
                    type="pil",
                    value=(
                        Image.open(Path("user_data/cache/pfp_me.png")) if Path("user_data/cache/pfp_me.png").exists() else None
                    ),
                    interactive=not mu,
                )

    with gr.Tab("Instruction template"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    dss_shared.gradio["instruction_template"] = gr.Dropdown(
                        choices=utils.get_available_instruction_templates(),
                        label="Saved instruction templates",
                        info='After selecting the template, click on "Load" to load and apply it.',
                        value="None",
                        elem_classes="slim-dropdown",
                    )
                    ui.create_refresh_button(
                        dss_shared.gradio["instruction_template"],
                        lambda: None,
                        lambda: {"choices": utils.get_available_instruction_templates()},
                        "refresh-button",
                        interactive=not mu,
                    )
                    dss_shared.gradio["load_template"] = gr.Button("Load", elem_classes="refresh-button")
                    dss_shared.gradio["save_template"] = gr.Button("💾", elem_classes="refresh-button", interactive=not mu)
                    dss_shared.gradio["delete_template"] = gr.Button("🗑️ ", elem_classes="refresh-button", interactive=not mu)

            with gr.Column():
                pass

        with gr.Row():
            with gr.Column():
                dss_shared.gradio["custom_system_message"] = gr.Textbox(
                    value=dss_shared.settings.get("custom_system_message", ""),
                    lines=2,
                    label="Custom system message (DSS)",
                    info="If not empty, will be used instead of the default one for this extension.",
                    elem_classes=["add_scrollbar"],
                )
                dss_shared.gradio["instruction_template_str"] = gr.Textbox(
                    value=dss_shared.settings.get("instruction_template_str", ""),
                    label="Instruction template (DSS)",
                    lines=24,
                    info="This gets autodetected or loaded. Used in instruct and chat-instruct modes for this extension.",
                    elem_classes=["add_scrollbar", "monospace"],
                )
                dss_shared.gradio["dss_instr_prompt_template"] = gr.Textbox(
                    value=dss_shared.settings.get("dss_instr_prompt_template", ""),
                    label="Response Instruction Prompt Template (DSS)",
                    lines=12,
                    info="Instruction prompt template for generating detailed instructions pre-reply.",
                    elem_classes=["add_scrollbar", "monospace"],
                    interactive=True,
                )
                with gr.Row():
                    dss_shared.gradio["send_instruction_to_default"] = gr.Button(
                        "Send to default", elem_classes=["small-button"]
                    )
                    dss_shared.gradio["send_instruction_to_notebook"] = gr.Button(
                        "Send to notebook", elem_classes=["small-button"]
                    )
                    dss_shared.gradio["send_instruction_to_negative_prompt"] = gr.Button(
                        "Send to negative prompt", elem_classes=["small-button"]
                    )

            with gr.Column():
                dss_shared.gradio["chat_template_str"] = gr.Textbox(
                    value=dss_shared.settings.get("chat_template_str", ""),
                    label="Chat template (DSS)",
                    lines=22,
                    info="Used in chat and chat-instruct modes for this extension.",
                    elem_classes=["add_scrollbar", "monospace"],
                )

        with gr.Row(elem_id="dss-chat-controls", elem_classes=["pretty_scrollbar"]):
            with gr.Column():
                with gr.Row():
                    dss_shared.gradio["start_with"] = gr.Textbox(
                        label="Start reply with",
                        placeholder="Sure thing!",
                        value=dss_shared.settings["start_with"],
                        elem_classes=["add_scrollbar"],
                    )

                with gr.Row():
                    dss_shared.gradio["mode"] = gr.Radio(
                        choices=["instruct", "chat-instruct", "chat"],
                        value=(
                            dss_shared.settings["mode"] if dss_shared.settings["mode"] in ["chat", "chat-instruct"] else None
                        ),
                        label="Mode",
                        info="Defines how the chat prompt is generated. In instruct and chat-instruct modes, the instruction template Parameters > Instruction template is used.",
                        elem_id="dss-chat-mode",
                    )

                with gr.Row():
                    dss_shared.gradio["chat-instruct_command"] = gr.Textbox(
                        value=dss_shared.settings["chat-instruct_command"],
                        lines=12,
                        label="Command for chat-instruct mode",
                        info="<|character|> and <|prompt|> get replaced with the bot name and the regular chat prompt respectively.",
                        visible=dss_shared.settings["mode"] == "chat-instruct",
                        elem_classes=["add_scrollbar"],
                    )


def create_block_ui():
    """Create the block UI under the chat tab."""
    with gr.Row():
        dss_shared.gradio["dss_toggle"] = gr.Checkbox(
            label="DAYNA Story Summarizer",
            value=dss_shared.persistent_ui_state.get("dss_toggle", True),
            interactive=True,
        )
    with gr.Row():
        dss_shared.gradio["count_tokens"] = gr.Button("Count tokens", size="sm")
    dss_shared.gradio["token_display"] = gr.HTML(value="", elem_classes="token-display")


def create_event_handlers():
    # Helper to update dss_shared.settings
    def update_dss_setting(key_name, value):
        dss_shared.settings[key_name] = value
        # print(f"DSS Setting {key_name} updated to: {value}")
        return value

    def update_dss_persistent_setting(key_name, value):
        dss_shared.persistent_ui_state[key_name] = value
        # print(f"DSS Persistent Setting {key_name} updated to: {value}")
        return value

    # Bind .change events for simple textboxes to update dss_shared.settings
    # Assuming the user has already updated the `value` attribute of these textboxes
    # to use dss_shared.settings.get('key_name', default_value)
    text_keys_to_bind = [
        "name1",
        "name2",
        "context",
        "greeting",
        "user_bio",
        "custom_system_message",
        "instruction_template_str",
        "chat_template_str",
        "dss_summarizer_instr_prompt_template",
    ]
    for key in text_keys_to_bind:
        if key in dss_shared.gradio:

            def callback(value, k=key):
                return update_dss_setting(k, value)

            dss_shared.gradio[key].change(fn=callback, inputs=dss_shared.gradio[key], outputs=None)

    dropdown_keys_to_bind = ["character_menu", "instruction_template"]
    for key in dropdown_keys_to_bind:
        if key in dss_shared.gradio:

            def callback(value, k=key):
                return update_dss_setting(k, value)

            dss_shared.gradio[key].change(fn=callback, inputs=dss_shared.gradio[key], outputs=None)

    # Save/delete a character
    dss_shared.gradio["save_character"].click(
        chat.handle_save_character_click,
        gradio("name2"),
        gradio("save_character_filename", "character_saver"),
        show_progress=False,
    )
    dss_shared.gradio["delete_character"].click(
        lambda: gr.update(visible=True), None, gradio("character_deleter"), show_progress=False
    )
    dss_shared.gradio["load_template"].click(
        chat.handle_load_template_click,
        gradio("instruction_template"),
        gradio("instruction_template_str", "instruction_template"),
        show_progress=False,
    )
    dss_shared.gradio["save_template"].click(
        utils.gather_interface_values, gradio(dss_shared.input_elements), gradio("interface_state")
    ).then(
        chat.handle_save_template_click,
        gradio("instruction_template_str"),
        gradio("save_filename", "save_root", "save_contents", "file_saver"),
        show_progress=False,
    )

    dss_shared.gradio["delete_template"].click(
        chat.handle_delete_template_click,
        gradio("instruction_template"),
        gradio("delete_filename", "delete_root", "file_deleter"),
        show_progress=False,
    )

    # Upload character handlers - These point to main app logic
    if all(k in dss_shared.gradio for k in ["Submit character", "upload_json", "upload_img_bot", "character_menu"]):
        dss_shared.gradio["Submit character"].click(
            chat.upload_character,
            gradio("upload_json", "upload_img_bot"),
            gradio("character_menu"),
            show_progress=False,
        ).then(None, None, None, js=f"() => {{{ui.switch_tabs_js}; switch_to_character()}}")

    if all(k in dss_shared.gradio for k in ["Submit tavern character", "upload_img_tavern", "tavern_json", "character_menu"]):
        dss_shared.gradio["Submit tavern character"].click(
            chat.upload_tavern_character,
            gradio("upload_img_tavern", "tavern_json"),
            gradio("character_menu"),
            show_progress=False,
        ).then(None, None, None, js=f"() => {{{ui.switch_tabs_js}; switch_to_character()}}")

    if all(k in dss_shared.gradio for k in ["upload_json", "Submit character"]):
        dss_shared.gradio["upload_json"].upload(lambda: gr.update(interactive=True), None, gradio("Submit character"))
        dss_shared.gradio["upload_json"].clear(lambda: gr.update(interactive=False), None, gradio("Submit character"))

    if all(
        k in dss_shared.gradio
        for k in [
            "upload_img_tavern",
            "tavern_name",
            "tavern_desc",
            "tavern_json",
            "Submit tavern character",
        ]
    ):
        dss_shared.gradio["upload_img_tavern"].upload(
            chat.check_tavern_character,
            gradio("upload_img_tavern"),
            gradio("tavern_name", "tavern_desc", "tavern_json", "Submit tavern character"),
            show_progress=False,
        )
        dss_shared.gradio["upload_img_tavern"].clear(
            lambda: (None, None, None, gr.update(interactive=False)),
            None,
            gradio("tavern_name", "tavern_desc", "tavern_json", "Submit tavern character"),
            show_progress=False,
        )

    dss_shared.gradio["character_menu"].change(
        utils.gather_interface_values, gradio(dss_shared.input_elements), gradio("interface_state")
    ).then(
        chat.load_character,
        gradio("character_menu", "name1", "name2"),
        gradio("name1", "name2", "character_picture", "greeting", "context"),
        show_progress=False,
    ).then(
        None, None, None, js=f"() => {{{ui.update_big_picture_js}; updateBigPicture()}}"
    )

    # Block
    dss_shared.gradio["count_tokens"].click(
        utils.gather_interface_values, gradio(dss_shared.input_elements), gradio("interface_state")
    ).then(
        chat.count_prompt_tokens,
        [shared.gradio["textbox"], dss_shared.custom_state],
        gradio("token_display"),
        show_progress=False,
    )

    dss_shared.gradio["dss_toggle"].change(
        utils.gather_interface_values, gradio(dss_shared.input_elements), gradio("interface_state")
    )
