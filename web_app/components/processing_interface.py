"""
Processing interface components for MVS web application
"""

import gradio as gr
from typing import Tuple, Optional
from constants import UI_LABELS


class ProcessingInterface:
    """Factory class for creating processing interface components"""
    
    @staticmethod
    def create_processing_controls() -> Tuple[gr.Button, gr.File, gr.Textbox]:
        """
        Create processing control components
        
        Returns:
            tuple: (process_pair_btn, download_ply, processing_status)
        """
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Processing")
                
                # Process pair button (disabled until model is loaded)
                process_pair_btn = gr.Button(
                    "🎯 Process Pair",
                    variant="secondary",
                    size="lg",
                    interactive=False
                )
                
                # Download PLY file
                download_ply = gr.File(
                    label="📥 Download Point Cloud",
                    visible=False,
                    interactive=False
                )
                
                # Processing status
                processing_status = gr.Textbox(
                    label="Status",
                    value="🔄 MASt3R model loading in background... Ready to process pairs once loaded.",
                    interactive=False,
                    lines=3
                )
                
                # Update status button
                update_status_btn = gr.Button(
                    "🔄 Update Status",
                    variant="secondary",
                    size="sm"
                )
                
        
        return process_pair_btn, download_ply, processing_status, update_status_btn
