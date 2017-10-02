Tutorial
========

.. contents::
	:local:
	:backlinks: none

Make simple sound
-----------------
This example is make 440 Hz sine wave sound.

.. doctest::

	>>> import gensound

	>>> sound = gensound.Sound.from_sinwave(440)

The :class:`Sound<gensound.sound.Sound>` class has some class methods for generating simple tones.
Please see :doc:`reference<modules/sound>`.

Play and save
-------------
Play made sound with :func:`play<gensound.sound.Sound.play>` method, like this.

.. code-block:: python

	>>> sound.play()

Or, can write into disk with :func:`write<gensound.sound.Sound.write>` method.

.. doctest::

	>>> sound.write('out.wav')

.. testcleanup::

	import os

	os.remove('out.wav')

Overlay or concatenate
----------------------
Make two sounds,

.. doctest::

	>>> a = gensound.Sound.from_sinwave(440)
	>>> b = gensound.Sound.from_sinwave(880)

and overlay they with :func:`overlay<gensound.sound.overlay>` function.

.. doctest::

	>>> overlay = gensound.overlay(a, b)

The ``overlay`` is the same duration as ``a`` and ``b``, and play both of 440 Hz and 880 Hz.

Or, concatenate they with :func:`concat<gensound.sound.concat>` function.

.. doctest::

	>>> concat = gensound.concat(a, b)

The ``concat`` is playing ``a`` then ``b``.

Use sound effects
-----------------
PyGenSound has some effects like :class:`fade-in<gensound.effect.LinearFadeIn>`, :class:`fade-out<gensound.effect.LinearFadeOut>`, :class:`high pass<gensound.effect.HighPassFilter>` or :class:`low pass filter<gensound.effect.LowPassFilter>`.
In PyGenSound, :class:`resampling<gensound.effect.Resampling>` and :class:`changing speed<gensound.effect.ChangeSpeed>` is classified as an effect.

This sample will apply :class:`fade-out<gensound.effect.LinearFadeOut>` effect to sound ``a``.

.. doctest::

	>>> effect = gensound.LinearFadeOut()
	>>> a_fadeout = effect.apply(a)

You can use effects as a like a stream operator of C++.

.. doctest::

	>>> a_fade = a >> gensound.LinearFadeOut() >> gensound.LinearFadeIn()

	>>> a_fade == gensound.LinearFadeIn() << gensound.LinearFadeOut() << a
	True

Please see detail about effects to :doc:`reference<modules/effect>`

Examples
--------
Make NHK time signal sound
""""""""""""""""""""""""""
.. code-block:: python

	import gensound


	wait = gensound.Sound.silence(duration=0.9)  # Generate 0.9 seconds silence
	a = gensound.Sound.from_sinwave(440, duration=0.1, volume=1.0).concat(wait)  # Generate 440Hz sin wave 0.1 seconds, and 0.9 seconds silence
	b = gensound.Sound.from_sinwave(880, duration=1.0, volume=1.0)  # Generate 880Hz sin wave 1 seconds

	time_signal = gensound.concat(a, a, a, b)  # Concatenate those

	time_signal.write('test.wav')  # Save to test.wav
	time_signal.play()  # Play sound
