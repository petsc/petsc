#!/usr/bin/env python3
"""
# Created: Wed Jul  5 15:40:46 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
class Color:
  try:
    import colorama # type: ignore[import]

    __COLOR_BRIGHT_RED__: str    = colorama.Fore.RED + colorama.Style.BRIGHT
    __COLOR_BRIGHT_YELLOW__: str = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    __COLOR_RESET__: str         = colorama.Style.RESET_ALL
  except ImportError:
    __COLOR_BRIGHT_RED__    = ''
    __COLOR_BRIGHT_YELLOW__ = ''
    __COLOR_RESET__         = ''

  @classmethod
  def bright_red(cls) -> str:
    r"""Return the ASCII code for bright red

    Returns
    -------
    ret :
      the ASCII code for bright red for the current terminal type
    """
    return cls.__COLOR_BRIGHT_RED__

  @classmethod
  def bright_yellow(cls) -> str:
    r"""Return the ASCII code for bright yellow

    Returns
    -------
    ret :
      the ASCII code for bright yellow
    """
    return cls.__COLOR_BRIGHT_YELLOW__

  @classmethod
  def reset(cls) -> str:
    r"""Return the ASCII code for resetting color

    Returns
    -------
    ret :
      the ASCII code to reset all color characteristics
    """
    return cls.__COLOR_RESET__
