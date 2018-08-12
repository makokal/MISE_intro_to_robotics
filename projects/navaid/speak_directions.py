import time
import pyttsx3

WARNING = 'Watch out!'
PAUSE = ' '
AHEAD = 'ahead'
LEFT = 'to the left'
RIGHT = 'to the right'


class  Speaker(object):
    """ A basic speaker for relaying navigational instructions to user."""
    def __init__(self):
        self._engine = pyttsx3.init()

    def say_direction(self, direction_text):
        # Make the sound.
        self._engine.say(direction_text)
        # self._engine.runAndWait()
        return True
        
    