import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# Create figure with 5 subplots in a row (landscape orientation)
fig, axs = plt.subplots(1, 5, figsize=(22, 12))
fig.subplots_adjust(wspace=0.4)

# Color scheme
colors = {
    'bert': '#4B4BFF',      # Blue
    'roberta': '#FF4B4B',   # Red
    'distilbert': '#FFB000', # Yellow/Orange
    'albert': '#00C9A7',    # Green
    'sbert': '#C47AFF',     # Purple
    'embedding': '#AAAAAA', # Gray
    'attention': '#72A0E5', # Light blue
    'ffn': '#90EE90',       # Light green 
    'pooling': '#FFD700',   # Gold
    'arrow': '#555555',     # Dark gray
}

# Standardized sizes for blocks - make them longer
STD_BLOCK_HEIGHT = 0.045
STD_BLOCK_WIDTH = 0.7

# Function to draw a transformer block
def draw_transformer_block(ax, y_pos, width=STD_BLOCK_WIDTH, height=STD_BLOCK_HEIGHT, color='gray', label=None, alpha=1.0, x_offset=0.15):
    rect = Rectangle((x_offset, y_pos), width, height, facecolor=color, edgecolor='black', alpha=alpha)
    ax.add_patch(rect)
    if label:
        ax.text(x_offset + width/2, y_pos + height/2, label, ha='center', va='center', fontsize=9)
    return y_pos + height

# Function to draw an arrow
def draw_arrow(ax, start_x, start_y, end_x, end_y, color=colors['arrow']):
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), 
                            arrowstyle='->', color=color, 
                            mutation_scale=15, linewidth=2)
    ax.add_patch(arrow)

# Function to create consistent annotation boxes with center-aligned position
def draw_annotation_box(ax, y_pos, text, center_x=0.5):
    # Use a standard fixed-width box, centered under each model
    ax.text(center_x, y_pos, text,
            va='center', ha='center', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', linewidth=1))

# 1. BERT - Standard 12 layers
ax = axs[0]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('BERT (110M parameters)', fontsize=12, fontweight='bold', color=colors['bert'])
ax.axis('off')

# Draw input
ax.text(0.5, 0.03, 'Input', ha='center', fontsize=10)

# Draw embedding layer
embedding_y = draw_transformer_block(ax, 0.06, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['embedding'], "Embedding Layer")

# Draw all 12 transformer blocks 
y_pos = embedding_y + 0.01
for i in range(12):
    y_pos = draw_transformer_block(ax, y_pos, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['bert'], f"Transformer Block {i+1}")

# Draw output
output_y = draw_transformer_block(ax, y_pos + 0.01, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['bert'], "Output Layer")

# Center-aligned annotation box
draw_annotation_box(ax, 0.94, "• 12 identical transformer blocks\n• 110M parameters\n• Bidirectional attention\n• 768 hidden dimensions\n• Pre-trained on masked language modeling")

# 2. RoBERTa - Same as BERT with different training
ax = axs[1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('RoBERTa (125M parameters)', fontsize=12, fontweight='bold', color=colors['roberta'])
ax.axis('off')

# Draw input
ax.text(0.5, 0.03, 'Input', ha='center', fontsize=10)

# Draw embedding layer
embedding_y = draw_transformer_block(ax, 0.06, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['embedding'], "Embedding Layer")

# Draw all 12 transformer blocks
y_pos = embedding_y + 0.01
for i in range(12):
    y_pos = draw_transformer_block(ax, y_pos, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['roberta'], f"Transformer Block {i+1}")

# Draw output
output_y = draw_transformer_block(ax, y_pos + 0.01, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['roberta'], "Output Layer")

# Center-aligned annotation box
draw_annotation_box(ax, 0.94, "• Same architecture as BERT\n• 125M parameters\n• Trained on 10x more data\n• Dynamic masking (different in each epoch)\n• Larger batch sizes (8K vs 256)")

# 3. DistilBERT - 6 layers, knowledge distillation
ax = axs[2]
# Expand the x-axis to make room for the teacher model box
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1)
ax.set_title('DistilBERT (66M parameters)', fontsize=12, fontweight='bold', color=colors['distilbert'])
ax.axis('off')

# Draw input
ax.text(0.5, 0.03, 'Input', ha='center', fontsize=10)

# Draw embedding layer
embedding_y = draw_transformer_block(ax, 0.06, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['embedding'], "Embedding Layer")

# Position teacher model within the visible area
teacher_x = 0.95
teacher_y = 0.06  # Same as embedding_y
teacher_width = 0.12
teacher_height = 0.5

# Draw BERT teacher model (simplified)
ax.add_patch(Rectangle((teacher_x, teacher_y), teacher_width, teacher_height, 
                     facecolor=colors['bert'], edgecolor='black', alpha=0.7))  # Using BERT blue color
ax.text(teacher_x + teacher_width/2, teacher_y + teacher_height/2, "BERT\nTeacher\nModel", 
        ha='center', va='center', fontsize=8, rotation=0)

# Draw student model (DistilBERT) - 6 layers
y_pos = embedding_y + 0.01
for i in range(6):
    y_pos = draw_transformer_block(ax, y_pos, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['distilbert'], f"Transformer Block {i+1}")
    
    # Draw knowledge distillation arrows for all blocks
    draw_arrow(ax, 0.15 + STD_BLOCK_WIDTH, y_pos - STD_BLOCK_HEIGHT/2, teacher_x, teacher_y + (i+0.5)*teacher_height/6, colors['distilbert'])

# Draw output
output_y = draw_transformer_block(ax, y_pos + 0.01, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['distilbert'], "Output Layer")

# Center-aligned annotation box
draw_annotation_box(ax, 0.94, "• 6 transformer blocks (half of BERT)\n• 66M parameters (40% less)\n• Knowledge distillation from BERT\n• Triple loss function to mimic teacher\n• 97% of BERT's performance")

# 4. ALBERT - Parameter sharing
ax = axs[3]
# Expand the x-axis to make room for the shared parameters box
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1)
ax.set_title('ALBERT (12M parameters)', fontsize=12, fontweight='bold', color=colors['albert'])
ax.axis('off')

# Draw input
ax.text(0.5, 0.03, 'Input', ha='center', fontsize=10)

# Make embedding boxes bigger for ALBERT's factorized embeddings
emb_width = STD_BLOCK_WIDTH/2 - 0.03
emb_x_offset1 = 0.15
emb_x_offset2 = 0.15 + emb_width + 0.06  # Add gap between boxes

# Draw factorized embedding (ALBERT's key innovation)
emb1_y = draw_transformer_block(ax, 0.06, emb_width, STD_BLOCK_HEIGHT, colors['embedding'], "Vocab→Low Dim", x_offset=emb_x_offset1)
emb2_y = draw_transformer_block(ax, 0.06, emb_width, STD_BLOCK_HEIGHT, colors['embedding'], "Low Dim→Hidden", x_offset=emb_x_offset2)

# Draw arrow between embeddings to show the two-step process
draw_arrow(ax, emb_x_offset1 + emb_width, 0.06 + STD_BLOCK_HEIGHT/2, emb_x_offset2, 0.06 + STD_BLOCK_HEIGHT/2, colors['arrow'])

# Position shared parameters box within the visible area
shared_params_x = 0.95
shared_params_y = 0.06  # Same as embedding_y
shared_params_width = 0.12
shared_params_height = 0.5

# Draw transformer blocks with parameter sharing
y_pos = emb1_y + 0.01

# Draw shared parameters box
ax.add_patch(Rectangle((shared_params_x, shared_params_y), shared_params_width, shared_params_height, 
                      facecolor=colors['albert'], edgecolor='black'))
ax.text(shared_params_x + shared_params_width/2, shared_params_y + shared_params_height/2, "Shared\nParameters\nAcross All\nLayers", ha='center', va='center', fontsize=8)

# Draw all 12 transformer blocks and connect all to shared parameters
for i in range(12):
    y_pos = draw_transformer_block(ax, y_pos, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['albert'], f"Transformer Block {i+1}")
    # Draw parameter sharing arrows for ALL blocks
    draw_arrow(ax, 0.15 + STD_BLOCK_WIDTH, y_pos - STD_BLOCK_HEIGHT/2, shared_params_x, shared_params_y + (i+0.5)*shared_params_height/12, colors['albert'])

# Draw output
output_y = draw_transformer_block(ax, y_pos + 0.01, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['albert'], "Output Layer")

# Center-aligned annotation box
draw_annotation_box(ax, 0.94, "• 12 transformer blocks with parameter sharing\n• Factorized embeddings (vocabulary→low dim→\n  hidden)\n• 12M parameters (89% less than BERT)\n• Same weights reused across all layers\n• Compact but computationally intensive")

# 5. SBERT - Sentence embeddings
ax = axs[4]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('SBERT (22M parameters)', fontsize=12, fontweight='bold', color=colors['sbert'])
ax.axis('off')

# Draw input
ax.text(0.5, 0.03, 'Input', ha='center', fontsize=10)

# Draw embedding layer
embedding_y = draw_transformer_block(ax, 0.06, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['embedding'], "Embedding Layer")

# Draw transformer blocks - 6 layers
y_pos = embedding_y + 0.01
for i in range(6):
    y_pos = draw_transformer_block(ax, y_pos, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['sbert'], f"Transformer Block {i+1}")

# Draw pooling layer
pooling_y = draw_transformer_block(ax, y_pos + 0.01, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['pooling'], "Mean Pooling Layer")

# Draw output
output_y = draw_transformer_block(ax, pooling_y + 0.01, STD_BLOCK_WIDTH, STD_BLOCK_HEIGHT, colors['sbert'], "Output Layer")

# Center-aligned annotation box
draw_annotation_box(ax, 0.94, "• 6 transformer blocks with sentence-level pooling\n• 22M parameters (based on MiniLM)\n• Optimized for creating sentence embeddings\n• Mean pooling creates fixed-size vectors\n• Fastest training time (144s)")

# Set a main title for the entire figure
fig.suptitle('BERT Model Variants: Architectural Comparison', fontsize=18, fontweight='bold', y=0.98)

# Save the figure
plt.savefig('bert_model_architectures_landscape.png', dpi=300, bbox_inches='tight')
plt.close()

print("Refined landscape architecture comparison image saved as 'bert_model_architectures_landscape.png'") 