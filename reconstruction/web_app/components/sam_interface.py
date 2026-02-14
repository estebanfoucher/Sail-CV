"""
SAM (Segment Anything Model) interface components for MVS web application
"""

import gradio as gr
from typing import Tuple, Optional
from constants import UI_LABELS


class SAMInterface:
    """Factory class for creating SAM interface components"""
    
    @staticmethod
    def create_sam_controls() -> Tuple[gr.Button, gr.Button, gr.Textbox, gr.Textbox, gr.Button]:
        """
        Create SAM control components
        
        Returns:
            tuple: (activate_sam_btn, deactivate_sam_btn, point_prompt_1, point_prompt_2, compute_masks_btn)
        """
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 SAM Filtering")
                
                # SAM activation button
                activate_sam_btn = gr.Button(
                    "🔍 Activate SAM Filter",
                    variant="primary",
                    size="md",
                    interactive=True
                )
                
                # SAM deactivation button
                deactivate_sam_btn = gr.Button(
                    "❌ Deactivate Filtering",
                    variant="secondary",
                    size="md",
                    interactive=False
                )
                
                # Point prompt inputs
                point_prompt_1 = gr.Textbox(
                    label="📍 Point Prompt - Image 1 (x,y)",
                    placeholder="e.g., 256,128",
                    info="Enter coordinates as 'x,y' for the point to segment in image 1",
                    interactive=False
                )
                
                point_prompt_2 = gr.Textbox(
                    label="📍 Point Prompt - Image 2 (x,y)",
                    placeholder="e.g., 256,128",
                    info="Enter coordinates as 'x,y' for the point to segment in image 2",
                    interactive=False
                )
                
                # Compute masks button
                compute_masks_btn = gr.Button(
                    "🎨 Compute Masks",
                    variant="secondary",
                    size="md",
                    interactive=False
                )
                
                # SAM status
                sam_status = gr.Textbox(
                    label="SAM Status",
                    value="🔴 SAM Filter: Inactive",
                    interactive=False,
                    lines=2
                )
        
        return activate_sam_btn, deactivate_sam_btn, point_prompt_1, point_prompt_2, compute_masks_btn, sam_status
