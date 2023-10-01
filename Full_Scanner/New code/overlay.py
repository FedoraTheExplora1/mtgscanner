import cv2
import numpy as np

def create_overlay():
    # Define the dimensions of the overlay
    overlay_width, overlay_height = 640, 480  # Adjust these values to match your webcam resolution

    # Create a transparent overlay image
    overlay = np.zeros((overlay_height, overlay_width, 4), dtype='uint8')

    # Define the positions and dimensions of the rectangles
    set_code_rect = [(10, overlay_height - 30), (60, overlay_height - 10)]
    name_rect = [(70, 10), (overlay_width - 10, 40)]

    # Draw the rectangles on the overlay
    cv2.rectangle(overlay, set_code_rect[0], set_code_rect[1], (0, 255, 0, 255), 2)  # Green rectangle for the set code
    cv2.rectangle(overlay, name_rect[0], name_rect[1], (0, 0, 255, 255), 2)  # Red rectangle for the card name

    # Save the overlay to a file
    cv2.imwrite('overlay.png', overlay)

    print(f'Overlay saved as overlay.png')

if __name__ == "__main__":
    create_overlay()
