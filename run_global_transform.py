import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # (height, width, 3)
    # Pad the image to avoid boundary issues,在图像周围添加白色背景蒙�?无论怎么绕中心转,都不会出�?
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    # Place the original image in the center of the new padded image
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转�?    

    # preparation
    # center of the image 
    h,w=image.shape[:2]
    center_x, center_y = w / 2, h / 2
    # scale : scale=1.5 <-> h=h*1.5, w=w*1.5
    # scale matrix
    S_scale=np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0, 0, 1]])
    # Translation to origin
    T_to_origin=np.array([[1, 0, -center_x],
                          [0, 1, -center_y],
                          [0, 0, 1]])
    T_back=np.array([[1, 0, center_x],
                     [0, 1, center_y],
                     [0, 0, 1]])
    scale_matrix=T_back @ S_scale @ T_to_origin


    # rotation : rotation=45 <-> rotate 45 degree clockwise
    # rotation matrix
    theta=np.deg2rad(rotation)
    R_rotation=np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    rotation_matrix=T_back @ R_rotation @ T_to_origin

    # translation : translation_x=100 <-> move right 100 pixels, translation_y=-50 <-> move up 50 pixels
    # translation matrix
    T_translation=np.array([[1, 0, translation_x],
                            [0, 1, translation_y],
                            [0, 0, 1]])
    translation_matrix=T_back @ T_translation @ T_to_origin

    # flip : flip_horizontal=True <-> flip horizontally, flip_horizontal=False <-> no flip
    # flip matrix
    if flip_horizontal:
        F_flip=np.array([[-1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    else:
        F_flip=np.eye(3)
    flip_matrix=T_back @ F_flip @ T_to_origin

    # composition matrix
    composition_matrix=flip_matrix @ translation_matrix @ rotation_matrix @ scale_matrix
    
    affine_matrix = composition_matrix[:2, :]
    transformed_image = cv2.warpAffine(image, 
                                       affine_matrix, (image.shape[1], image.shape[0]), 
                                       borderValue=(255,255,255))
    
    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
