import streamlit as st
import subprocess
import os
from pathlib import Path
import logging
import subprocess
from typing import Optional
import signal

def run_stylegan2_command(command: str) -> Optional[str]:
    """Run StyleGAN2 command and display logs"""
    try:
        # Create placeholders
        log_placeholder = st.empty()
        stop_placeholder = st.empty()
        
        # Add stop button
        stop_clicked = stop_placeholder.button("Stop Training")
        
        # Run command and capture output
        process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Display logs in real-time
        output = []
        while True:
            if stop_clicked:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                st.warning("Training stopped by user")
                break
                
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                output.append(line)
                # Show only last 50 lines
                display_output = output[-50:] if len(output) > 50 else output
                log_placeholder.markdown(
                    f"""<div class="log-container">
                    {"".join(display_output)}
                    </div>""",
                    unsafe_allow_html=True
                )
        
        # Wait for process to complete
        process.wait()
        
        # Remove stop button after completion
        stop_placeholder.empty()
        
        if process.returncode == 0:
            st.success("Generation completed successfully!")
            return "".join(output)
        else:
            st.error(f"Generation failed with return code {process.returncode}")
            return None
            
    except Exception as e:
        st.error(f"Error running StyleGAN2: {str(e)}")
        logging.exception("StyleGAN2 error")
        return None

st.title("StyleGAN2 Training Interface")
st.markdown("""
    <style>
        .log-container {
            height: 400px;
            overflow-y: auto;
            background-color: black;
            color: white;
            padding: 10px;
            font-family: monospace;
            white-space: pre;
        }
    </style>
""", unsafe_allow_html=True)
# Main tabs
tab1, tab2 = st.tabs(["Training", "Generation"])

with tab1:
    st.header("Training Configuration")
    
    # Basic Settings
    st.subheader("Basic Settings")
    data_path = st.text_input("Data Directory Path", "")
    project_name = st.text_input("Project Name", "default")
    
    col1, col2 = st.columns(2)
    with col1:
        image_size = st.number_input("Image Size", min_value=64, max_value=1024, value=128, step=64)
        batch_size = st.number_input("Batch Size", min_value=1, value=16)
    with col2:
        network_capacity = st.number_input("Network Capacity", min_value=1, value=16)
        num_train_steps = st.number_input("Training Steps", min_value=1000, value=150000, step=1000)

    # Advanced Settings
    st.subheader("Advanced Settings")
    
    col3, col4 = st.columns(2)
    with col3:
        gradient_accumulate = st.number_input("Gradient Accumulation", min_value=1, value=1)
        results_dir = st.text_input("Results Directory", "results")
    with col4:
        models_dir = st.text_input("Models Directory", "models")
        transparent = st.checkbox("Train on Transparent Images")

    # Augmentation Settings
    st.subheader("Augmentation Settings")
    
    aug_prob = st.slider("Augmentation Probability", 0.0, 1.0, 0.0, 0.05)
    aug_types = st.multiselect("Augmentation Types", 
                              ["translation", "cutout", "color"], 
                              default=[])

    # Attention Settings
    st.subheader("Attention Settings")
    attn_layers = st.multiselect("Attention Layers", 
                                [1, 2, 3, 4], 
                                default=[])

    # Multi-GPU Settings
    st.subheader("Multi-GPU Settings")
    multi_gpus = st.checkbox("Use Multiple GPUs")

    if st.button("Start Training"):
        if not data_path:
            st.error("Please specify the data directory path!")
        else:
            command = f"stylegan2_pytorch --data {data_path}"
            
            # Add basic settings
            command += f" --name {project_name}"
            command += f" --image-size {image_size}"
            command += f" --batch-size {batch_size}"
            command += f" --network-capacity {network_capacity}"
            command += f" --num-train-steps {num_train_steps}"
            
            # Add advanced settings
            command += f" --gradient-accumulate-every {gradient_accumulate}"
            command += f" --results_dir {results_dir}"
            command += f" --models_dir {models_dir}"
            
            if transparent:
                command += " --transparent"
                
            # Add augmentation settings
            if aug_prob > 0:
                command += f" --aug-prob {aug_prob}"
                if aug_types:
                    aug_types_str = "[" + ",".join(aug_types) + "]"
                    command += f" --aug-types {aug_types_str}"
                    
            # Add attention settings
            if attn_layers:
                attn_layers_str = "[" + ",".join(map(str, attn_layers)) + "]"
                command += f" --attn-layers {attn_layers_str}"
                
            # Add multi-GPU setting
            if multi_gpus:
                command += " --multi-gpus"
                
            st.code(command)
            run_stylegan2_command(command)

with tab2:
    st.header("Image Generation")
    
    generation_type = st.radio("Generation Type", 
                              ["Single Images", "Interpolation"])
    
    checkpoint = st.number_input("Load From Checkpoint (optional)", 
                               min_value=0, value=0)
    
    trunc_psi = st.slider("Truncation Psi", 0.0, 1.0, 0.75, 0.05)
    
    if generation_type == "Interpolation":
        num_steps = st.number_input("Interpolation Steps", 
                                  min_value=2, value=100)
        save_frames = st.checkbox("Save Individual Frames")
    
    if st.button("Generate"):
        command = "stylegan2_pytorch --generate"
        
        if generation_type == "Interpolation":
            command = "stylegan2_pytorch --generate-interpolation"
            command += f" --interpolation-num-steps {num_steps}"
            if save_frames:
                command += " --save-frames"
        
        if checkpoint > 0:
            command += f" --load-from {checkpoint}"
            
        command += f" --trunc-psi {trunc_psi}"
        
        st.code(command)
        run_stylegan2_command(command)

st.sidebar.header("Memory Usage Tips")
st.sidebar.markdown("""
* Decrease batch size and increase gradient accumulation for limited memory
* Lower network capacity can reduce memory requirements
* For high-resolution images (1024x1024), 16GB+ GPU memory is recommended
* Consider using 'Lightweight' GAN for better memory efficiency
""") 