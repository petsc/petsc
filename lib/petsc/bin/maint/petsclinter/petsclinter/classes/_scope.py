#!/usr/bin/env python3
"""
# Created: Tue Aug  8 09:21:41 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

class Scope:
  """
  Scope encompasses both the logical and lexical reach of a callsite, and is used to
  determine if two function calls may be occur in chronological order. Scopes may be
  approximated by incrementing or decrementing a counter every time a pair of '{}' are
  encountered however it is not that simple. In practice they behave almost identically
  to sets. Every relation between scopes may be formed by the following axioms.

  - Scope A is said to be greater than scope B if one is able to get to scope B from scope A
  e.g.:
  { // scope A
    { // scope B < scope A
      ...
    }
  }
  - Scope A is said to be equivalent to scope B if and only if they are the same object.
  e.g.:
  { // scope A and scope B
    ...
  }

  One notable exception are switch-case statements. Here every 'case' label acts as its
  own scope, regardless of whether a "break" is inserted i.e.:

  switch (cond) { // scope A
  case 1: // scope B begin
    ...
    break; // scope B end
  case 2: // scope C begin
    ...
  case 2:// scope C end, scope D begin
    ...
    break; // scope D end
  }

  Semantics here are weird, as:
  - scope B, C, D < scope A
  - scope B != scope C != scope D
  """
  __slots__ = ('children',)

  children: list[Scope]

  def __init__(self) -> None:
    r"""Construct a `Scope`"""
    self.children = []
    return

  def __lt__(self, other: Scope) -> bool:
    assert isinstance(other, Scope)
    return not self >= other

  def __gt__(self, other: Scope) -> bool:
    assert isinstance(other, Scope)
    return self.is_child_of(other)

  def __le__(self, other: Scope) -> bool:
    assert isinstance(other, Scope)
    return not self > other

  def __ge__(self, other: Scope) -> bool:
    assert isinstance(other, Scope)
    return (self > other) or (self == other)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Scope):
      return NotImplemented
    return id(self) == id(other)

  def __ne__(self, other: object) -> bool:
    return not self == other

  def sub(self) -> Scope:
    r"""Create a sub-scope

    Returns
    -------
    child :
      the sub-scope, which is automatically a child of `self`
    """
    child = Scope()
    self.children.append(child)
    return child

  def is_parent_of(self, other: Scope) -> bool:
    r"""Determine whether `self` is parent of `other`

    Parameters
    ----------
    other :
      the other `Scope`

    Returns
    -------
    ret :
      True if `self` is a lexical parent of `other`, False otherwise
    """
    if self == other:
      return False
    for child in self.children:
      if (other == child) or child.is_parent_of(other):
        return True
    return False

  def is_child_of(self, other: Scope) -> bool:
    r"""Determine whether `self` is child of `other`

    Parameters
    ----------
    other :
      the other `Scope`

    Returns
    -------
    ret :
      True if `self` is a lexical child of `other`, False otherwise
    """
    return other.is_parent_of(self)
