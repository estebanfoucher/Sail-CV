"""
MVS Interactive Web Application - Phase 1
Basic video loading and single frame processing
"""

import gradio as gr
import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import configuration and services
from config.settings import setup_logging, get_config
from components.ui_factory import UIFactory
from services.event_handler import EventHandler

# Setup logging
logger = setup_logging()
config = get_config()


def create_mvs_app():
    """Create the main MVS Gradio application"""
    
    app_config = config['app']
    ui_config = config['ui']
    
    with gr.Blocks(
        title=app_config['title'],
        theme=getattr(gr.themes, app_config['theme'].title())(),
        css=f"""
        .container {{ max-width: {ui_config['container_max_width']}; margin: 0 auto; }}
        .upload-section {{ margin: {ui_config['upload_section_margin']}; }}
        .status-box {{ font-family: monospace; }}
        """
    ) as app:
        
        gr.Markdown("# Multi-View Stereo 3D Reconstruction Interactive Web Application")
        gr.Markdown("## 🎬 Video Upload")
        
        # Create UI components using factory
        components = UIFactory.create_complete_interface()
        
        # Video Upload and Display Section
        with gr.Row():
            with gr.Column(scale=2):
                # Video upload and display components
                components['video_1_upload']
                components['video_1_display']
            with gr.Column(scale=2):
                # Video upload and display components
                components['video_2_upload']
                components['video_2_display']
        
        # Calibration Upload Section
        with gr.Row():
            with gr.Column(scale=1):
                components['calibration_upload']
            with gr.Column(scale=1):
                components['upload_status']
        
        # Frame Selection Section
        with gr.Row():
            with gr.Column(scale=1):
                # Frame selection controls
                components['frame_slider']
        
        # Selected Frames Section
        with gr.Row():
            with gr.Column(scale=1):
                # Selected frames display
                components['selected_image_1']
            with gr.Column(scale=1):
                # Selected frames display
                components['selected_image_2']
        
        # Processing Section
        with gr.Row():
            with gr.Column(scale=1):
                components['process_pair_btn']
                components['render_camera_toggle']
                components['download_ply']
                components['processing_status']
        
        # 3D Visualization Section
        with gr.Row():
            with gr.Column(scale=2):
                components['model3d_viewer']
            with gr.Column(scale=1):
                components['viz_status']
        
        # Batch Processing Section
        with gr.Row():
            with gr.Column(scale=1):
                components['folder_name']
                components['start_frame']
                components['end_frame']
                components['step']
                components['process_all_btn']
                components['batch_progress']
                components['batch_status']
                components['download_batch']
        
        # Setup event handlers
        event_handler = EventHandler()
        event_handler.setup_event_handlers(components)
    
    return app


def main():
    """Main function to run the application"""
    try:
        logger.info("Starting MVS Interactive Web Application")
        
        # Create the application
        app = create_mvs_app()
        
        # Launch the application
        app_config = config['app']
        app.launch(
            server_port=app_config['server_port'],
            server_name=app_config['server_name'],
            share=app_config['share'],
            debug=app_config['debug']
        )
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()