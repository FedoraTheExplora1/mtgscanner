import cv2
import json

# Global variables to store the coordinates of the rectangles
rect_pts = []
drawing = False
img = None
boxes = {
    "set_code": None,
    "card_name": None
}
current_box = "set_code"

def draw_rectangle(event, x, y, flags, param):
    global rect_pts, drawing, img, boxes, current_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_pts = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, rect_pts[0], (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_pts.append((x, y))
        boxes[current_box] = rect_pts
        cv2.rectangle(img, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
        cv2.imshow('Image', img)

def save_template():
    with open('template.json', 'w') as f:
        json.dump(boxes, f, indent=4)

def switch_box():
    global current_box
    current_box = "card_name" if current_box == "set_code" else "set_code"

def main():
    global img
    img = cv2.imread(r'C:\Users\conno\Downloads\znr-120-pelakka-predation.jpg')  # Replace with the path to your MTG card image
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', draw_rectangle)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('s'):  # Press 's' to save the template
            save_template()
        elif key == ord('c'):  # Press 'c' to switch between drawing the set code and card name boxes
            switch_box()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
