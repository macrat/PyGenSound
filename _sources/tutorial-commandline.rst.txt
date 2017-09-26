Commandline Tutorial
====================

This document is about using PyGenSound from command line.

Please see :doc:`another tutorial<tutorial>` if you want to use from python.

.. contents::
	:local:
	:backlinks: none

Make simple sound
-----------------
This example is make 440 Hz sine wave sound and write to ``out.wav``.

.. code-block:: shell-session

	$ gensound sine 440 -o out.wav

Play sound without save
-----------------------
Play made sound without saving to disk, with mplayer.

.. code-block:: shell-session

	$ gensound sine 440 | mplayer -cache 1024 -

Overlay or concatenate
----------------------
Make two sounds,

.. code-block:: shell-session

	$ gensound sine 440 -o 440.wav
	$ gensound sine 880 -o 880.wav

and overlay they.

.. code-block:: shell-session

	$ gensound overlay 440.wav 880.wav -o overlay.wav

The ``overlay.wav`` is the same duration as ``440.wav`` and ``880.wav``, and play both of 440 Hz and 880 Hz.

Or, concatenate they.

.. code-block:: shell-session

	$ gensound concat 440.wav 880.wav -o concat.wav

The ``concat.wav`` is playing ``440.wav`` then ``880.wav``.

Use sound effects
-----------------
PyGenSound has some effects like fade-in, fade-out, high pass or low pass filter.

This sample will apply fade-in effect to sound ``440.wav``.

.. code-block:: shell-session

	$ gensound fadeout -i 440.wav -o fadeout.wav

Examples
--------
Make NHK time signal sound
""""""""""""""""""""""""""
.. code-block:: shell-session

	$ gensound silence -d 0.9 -o silence.wav
	$ gensound sine 440 -d 0.1 | gensound fadeout -o 440.wav
	$ gensound sine 880 -d 2.0 | gensound fadeout -o 880.wav
	$ gensound concat 440.wav silence.wav | gensound concat - - - 880.wav | mplayer -cache 1024 -
