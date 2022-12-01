# import keyboard.virtual_keyboard
import testing
from keyboard import virtual_keyboard
from drawing import LiveFeedDraw
from presentation_module import presenter
from tkinter import *
from language_learner import learner

BACKGROUND = "#1a1a2e"
TEXT_BOX_BG = "#0f3460"
FONT_NAME = "Calibri"

window = Tk()

window.title("COMPUTER VISION BOARD")
window.config(padx=150, pady=100, bg=BACKGROUND)
canvas = Canvas(width=200, height=224, bg=BACKGROUND, highlightthickness=0)


iconImage = PhotoImage(file="project_logo.png")
canvas.create_image(100, 100, image=iconImage, )
canvas.grid(row=0, column=1)
title_label = Label(text="COMPUTER VISION BOARD", bg=BACKGROUND, fg="white", font=(FONT_NAME, 22, "bold"))
title_label.config(padx=10, pady=10)

keyboard_button = Button(text="Use Keyboard", highlightthickness=0, font=(FONT_NAME, 11, "bold"),
                         borderwidth=2, background="#d4483b", fg="white", activebackground="#d4483b",
                         activeforeground="white", width=15, command=virtual_keyboard.run_keyboard)
draw_button = Button(text="Use Drawer", highlightthickness=0, font=(FONT_NAME, 11, "bold"),
                     borderwidth=2, background="#d4483b", fg="white", activebackground="#d4483b",
                     activeforeground="white", width=15, command=LiveFeedDraw.run_drawer)
presenter_button = Button(text="Use Presenter", highlightthickness=0, font=(FONT_NAME, 11, "bold"),
                          borderwidth=2, background="#d4483b", fg="white", activebackground="#d4483b",
                          activeforeground="white", width=15, command=presenter.run_presenter)
learning_button = Button(text="Use Learner", highlightthickness=0, font=(FONT_NAME, 11, "bold"),
                         borderwidth=2, background="#d4483b", fg="white", activebackground="#d4483b",
                         activeforeground="white", width=15, command=learner.run_learner)
exit_button = Button(window, text="Exit", highlightthickness=0, font=(FONT_NAME, 11, "bold"),
                     borderwidth=2, background="#d4483b", fg="white", activebackground="#d4483b",
                     activeforeground="white", width=15, command=window.destroy)

# canvas.grid(column=0, row=0)
title_label.grid(row=0, column=1)

keyboard_button.grid(row=1, column=0)
draw_button.grid(row=1, column=2)
presenter_button.grid(row=2, column=0)
# exit_button.grid(row=2, column=1)
learning_button.grid(row=2, column=2)
window.mainloop()
