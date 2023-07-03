#!/usr/bin/env python3
"""
# Created: Wed Jul  5 15:40:46 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
class Color:
  try:
    import colorama

    __COLOR_BRIGHT_RED__    = colorama.Fore.RED + colorama.Style.BRIGHT
    __COLOR_BRIGHT_YELLOW__ = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    __COLOR_RESET__         = colorama.Style.RESET_ALL
  except ImportError:
    __COLOR_BRIGHT_RED__    = ''
    __COLOR_BRIGHT_YELLOW__ = ''
    __COLOR_RESET__         = ''

  @classmethod
  def bright_red(cls):
    return cls.__COLOR_BRIGHT_RED__

  @classmethod
  def bright_yellow(cls):
    return cls.__COLOR_BRIGHT_YELLOW__

  @classmethod
  def reset(cls):
    return cls.__COLOR_RESET__


color = Color()
