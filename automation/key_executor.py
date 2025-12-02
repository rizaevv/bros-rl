from pynput.keyboard import Key, Controller
import time

class KeyExecutor:
    def __init__(self):
        self.keyboard = Controller()
        self.key_map = {
            "left": Key.left,
            "right": Key.right,
            "up": Key.up,
            "down": Key.down,
            "space": Key.space,
            "enter": Key.enter,
            "esc": Key.esc
        }

    def press_key(self, key_name, duration=0.01):
        """Presses a key for a specified duration."""
        if key_name not in self.key_map:
            print(f"Key {key_name} not found in key_map.")
            return

        key = self.key_map[key_name]
        self.keyboard.press(key)
        time.sleep(duration)
        self.keyboard.release(key)

    def hold_key(self, key_name):
        """Holds a key down."""
        if key_name in self.key_map:
            self.keyboard.press(self.key_map[key_name])

    def release_key(self, key_name):
        """Releases a key."""
        if key_name in self.key_map:
            self.keyboard.release(self.key_map[key_name])
