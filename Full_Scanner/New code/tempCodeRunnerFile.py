    ret, frame = self.vid.read()
        if ret:
            adjusted_frame = self.adjust_brightness_and_contrast(frame)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)