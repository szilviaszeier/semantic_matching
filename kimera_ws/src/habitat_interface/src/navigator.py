#!/usr/bin/env python
from tkinter import Tk, Canvas
import rospy
from std_msgs.msg import String

class Navigator(object):
    def __init__(self):
        super().__init__()
        rospy.init_node('navigator')
        self.init_controls()
        self.publisher = rospy.Publisher('navigator/action', String, queue_size=1)
        self.publisher.publish("stay")
        rospy.on_shutdown(self.window.destroy)
        self.window.mainloop()


    def init_controls(self):
        self.window = Tk()  # Create a window
        self.window.title("Arrow Keys")  # Set a title

        self.canvas = Canvas(self.window, bg="white", width=300, height=300)
        self.canvas.pack()

        # Bind canvas with key events
        self.canvas.bind("<Up>", self.action_up)
        self.canvas.bind("<Down>", self.action_down)
        self.canvas.bind("<Left>", self.action_left)
        self.canvas.bind("<Right>", self.action_right)
        self.canvas.bind('<Escape>', self.action_exit)
        self.canvas.bind('<space>', self.action_stay)
        self.canvas.bind('<Control-Up>', self.action_look_up)
        self.canvas.bind('<Control-Down>', self.action_look_down)
        self.canvas.bind('<Shift-Up>', self.action_move_up)
        self.canvas.bind('<Shift-Down>', self.action_move_down)
        self.canvas.focus_set()

        self.window.protocol("WM_DELETE_WINDOW", lambda:self.action_exit("<Close>"))

    def action_up(self, event):
        self.publisher.publish('move_forward')

    def action_down(self, event):
        self.publisher.publish('move_backward')

    def action_left(self, event):
        self.publisher.publish('turn_left')

    def action_stay(self, event):
        self.publisher.publish('stay')

    def action_right(self, event):
        self.publisher.publish('turn_right')

    def action_exit(self, event):
        self.publisher.publish('exit')
        self.window.quit()

    def action_look_up(self, event):
        self.publisher.publish('look_up')

    def action_look_down(self, event):
        self.publisher.publish('look_down')

    def action_move_up(self, event):
        self.publisher.publish('move_up')

    def action_move_down(self, event):
        self.publisher.publish('move_down')

if __name__ == "__main__":
    Navigator()