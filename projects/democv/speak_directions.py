import time

from gtts import gTTS
from pygame import mixer

WARNING = 'Watch out!'
PAUSE = ' '
AHEAD = 'ahead'
LEFT = 'to the left'
RIGHT = 'to the right'


class  Speaker(object):
    """ A basic speaker for relaying navigational instructions to user."""
    def __init__(self):
        mixer.init()

    def say_direction(self, direction_text):
        # Make the sound.
        tts = gTTS(text=direction_text, lang='en')
        tts.save('talk_output.mp3')

        # Play the sound.
        mixer.music.load('talk_output.mp3')
        mixer.music.play()

    

if __name__ == '__main__':
    speaker = Speaker()

    speaker.say_direction(WARNING + PAUSE + ' there is an object ' + AHEAD)

    time.sleep(1)  # Pause a bit.
    speaker.say_direction(WARNING + PAUSE + ' there is an object ' + RIGHT)
