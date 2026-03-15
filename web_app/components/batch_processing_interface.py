"""
Batch processing interface components for MVS web application
"""

import gradio as gr
from typing import Tuple, Optional
from constants import UI_LABELS


class BatchProcessingInterface:
    """Factory class for creating batch processing interface components"""

    @staticmethod
    def create_batch_processing_controls() -> Tuple[gr.Textbox, gr.Number, gr.Number, gr.Number,
                                                   gr.Button, gr.Textbox, gr.Textbox, gr.File, gr.Button]:
        """
        Create batch processing control components

        Returns:
            tuple: (folder_name, start_frame, end_frame, step, process_all_btn,
                   batch_progress, batch_status, download_batch, update_batch_status_btn)
        """
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎬 Batch Processing")

                # Folder name input
                folder_name = gr.Textbox(
                    label=UI_LABELS['folder_name'],
                    value="full_video",
                    placeholder="Enter folder name for output",
                    interactive=True
                )

                # Frame range controls
                with gr.Row():
                    with gr.Column(scale=1):
                        start_frame = gr.Number(
                            label=UI_LABELS['start_frame'],
                            value=0,
                            minimum=0,
                            step=1,
                            interactive=True
                        )

                    with gr.Column(scale=1):
                        end_frame = gr.Number(
                            label=UI_LABELS['end_frame'],
                            value=100,
                            minimum=1,
                            step=1,
                            interactive=True
                        )

                    with gr.Column(scale=1):
                        step = gr.Number(
                            label=UI_LABELS['step'],
                            value=1,
                            minimum=1,
                            step=1,
                            interactive=True
                        )

                # Process all frames button
                process_all_btn = gr.Button(
                    UI_LABELS['process_all_btn'],
                    variant="primary",
                    size="lg",
                    interactive=False
                )

                # Progress display
                batch_progress = gr.Textbox(
                    label=UI_LABELS['batch_progress'],
                    value="Ready to process frames...",
                    interactive=False,
                    lines=2
                )

                # Batch processing status
                batch_status = gr.Textbox(
                    label=UI_LABELS['batch_status'],
                    value="Ready to process frames...",
                    interactive=False,
                    lines=4
                )

                # Download batch results
                download_batch = gr.File(
                    label=UI_LABELS['download_batch'],
                    visible=False,
                    interactive=False
                )

                # Update batch status button
                update_batch_status_btn = gr.Button(
                    "🔄 Update Batch Status",
                    variant="secondary",
                    size="sm"
                )

        return folder_name, start_frame, end_frame, step, process_all_btn, batch_progress, batch_status, download_batch, update_batch_status_btn
