    def capture_card(self):
        ret, frame = self.cap.read()
        if ret:
            contours = find_card(frame)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                self.card = get_card_perspective(frame, contour)
                if self.card is not None:
                    self.status_var.set("Card captured. Press 'Scan Card' to identify.")
                    self.display_card()  # Display the captured card
                else:
                    self.status_var.set("Failed to get card perspective.")
                    messagebox.showwarning("Error", "Failed to get card perspective.")
            else:
                self.status_var.set("No card detected. Try again.")
        else:
            self.status_var.set("Failed to capture the image.")
            messagebox.showwarning("Error", "Failed to capture the image.")
        if hasattr(self, 'card') and self.card is not None:
            print("Scanning card...")
            self.status_var.set("Scanning card...")
            self.update()
            try:
                card_hash = compute_card_hash(self.card)
                match = find_closest_match(self.conn, card_hash)
                if match:
                    print(f"Card identified: {match}")
                    self.match = match  # Set the self.match attribute
                    self.status_var.set(f"Card identified: {match}")
                    messagebox.showinfo("Card Identified", f"Card Name: {match}")
                else:
                    print("Card not recognized.")
                    self.status_var.set("Card not recognized.")
                    messagebox.showwarning("Card Not Recognized", "The card could not be identified.")
            except Exception as e:
                print(f"An error occurred: {e}")
                self.status_var.set(f"An error occurred: {e}")
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                cv2.destroyAllWindows()
        else:
            self.status_var.set("Failed to get card perspective.")  # Added this line to set status if card perspective is None