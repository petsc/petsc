#!/usr/bin/env python3
"""
# Created: Tue Jun 21 09:25:37 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import shutil
import difflib
import tempfile
import traceback
import petsclinter as pl

from .main          import ReturnCode
from .classes._path import Path
from .util._utility import traceback_format_exception

from ._typing import *

class TemporaryCopy:
  __slots__ = 'fname', 'tmp', 'temp_path'

  def __init__(self, fname: Path) -> None:
    self.fname = fname.resolve(strict=True)
    return

  def __enter__(self) -> TemporaryCopy:
    self.tmp       = tempfile.NamedTemporaryFile(suffix=self.fname.suffix)
    self.temp_path = Path(self.tmp.name).resolve(strict=True)
    shutil.copy2(str(self.fname), str(self.temp_path))
    return self

  def __exit__(self, *args, **kwargs) -> None:
    self.orig_file().unlink(missing_ok=True)
    self.rej_file().unlink(missing_ok=True)
    del self.tmp
    del self.temp_path
    return

  def orig_file(self) -> Path:
    return self.temp_path.append_suffix('.orig')

  def rej_file(self) -> Path:
    return self.temp_path.append_suffix('.rej')

def test_main(
    petsc_dir:    Path,
    test_path:    Path,
    output_dir:   Path,
    patch_list:   list[PathDiffPair],
    errors_fixed: list[CondensedDiags],
    errors_left:  list[CondensedDiags],
    replace:      bool = False,
) -> ReturnCode:
  r"""The "main" function for testing

  Parameters
  ----------
  petsc_dir :
    the path to $PETSC_DIR
  test_path :
    the path to test files
  output_dir :
    the path containing all of the output against which the generated output is compared to
  patch_list :
    the list of generated patches
  errors_fixed :
    the set of generated (but fixed) errors
  errors_left :
    the set of generated (and not fixed) errors
  replace :
    should the output be replaced?

  Returns
  -------
  ret :
    `ReturnCode.ERROR_TEST_FAILED` if generated output does not match expected, and `ReturnCode.SUCCESS`
    otherwise
  """
  def test(generated_output: list[str], reference_file: Path) -> str:
    short_ref_name = reference_file.relative_to(petsc_dir)
    if replace:
      pl.sync_print('\tREPLACE', short_ref_name)
      reference_file.write_text(''.join(generated_output))
      return ''
    if not reference_file.exists():
      return f'Missing reference file \'{reference_file}\'\n'
    return ''.join(
      difflib.unified_diff(
        reference_file.read_text().splitlines(True), generated_output,
        fromfile=str(short_ref_name), tofile='Generated Output', n=0
      )
    )

  # sanitize the output so that it will be equal across systems
  def sanitize_output_file(text: Optional[str]) -> list[str]:
    return [] if text is None else [l.replace(str(petsc_dir), '.') for l in text.splitlines(True)]

  def sanitize_patch_file(text: Optional[str]) -> list[str]:
    # skip the diff header with file names
    return [] if text is None else text.splitlines(True)[2:]

  def rename_patch_file_target(text: str, new_path: Path) -> str:
    lines    = text.splitlines(True)
    out_file = lines[0].split()[1]
    lines[0] = lines[0].replace(out_file, str(new_path))
    lines[1] = lines[1].replace(out_file, str(new_path))
    return ''.join(lines)

  FIXED_MARKER = '<--- FIXED --->'
  LEFT_MARKER  = '<--- LEFT --->'
  patch_error = {}
  root_dir    = f'--directory={petsc_dir.anchor}'
  patches     = dict(patch_list)

  tmp_output      = {
    p : [FIXED_MARKER, '\n'.join(s), LEFT_MARKER]
    for diags in errors_fixed
      for p, s in diags.items()
  }
  for diags in errors_left:
    for path, strlist in diags.items():
      if path not in tmp_output:
        tmp_output[path] = [f'{FIXED_MARKER}\n{LEFT_MARKER}']
      tmp_output[path].extend(strlist)

  # ensure that each output ends with a newline
  output = {
    path : '\n'.join(strlist if strlist[-1].endswith('\n') else strlist + [''])
    for path, strlist in tmp_output.items()
  }
  # output = {
  #   path : '\n'.join(strlist if len(strlist) == 4 else strlist + ['']) for path, strlist in tmp_output.items()
  # }
  #output = {key : '\n'.join(val if len(val) == 4 else val + ['']) for key, val in output.items()}
  if test_path.is_dir():
    c_suffixes = (r'*.c', r'*.cxx', r'*.cpp', r'*.cc', r'*.CC')
    file_list  = [item for sublist in map(test_path.glob, c_suffixes) for item in sublist]
  else:
    file_list  = [test_path]
  for test_file in file_list:
    output_base = output_dir / test_file.stem
    output_file = output_base.with_suffix('.out')
    patch_file  = output_base.with_suffix('.patch')
    short_name  = test_file.relative_to(petsc_dir)

    pl.sync_print('\tTEST   ', short_name)

    output_errors = [
      test(sanitize_output_file(output.get(test_file)), output_file),
      test(sanitize_patch_file(patches.get(test_file)), patch_file)
    ]

    # no point in checking the patch, we have already replaced
    if not replace:
      # make sure the patch can be applied
      with TemporaryCopy(test_file) as tmp_src, \
           tempfile.NamedTemporaryFile(delete=True, suffix='.patch') as temp_patch:
        tmp_patch_path = Path(temp_patch.name).resolve(strict=True)
        try:
          tmp_patch_path.write_text(rename_patch_file_target(patches[test_file], tmp_src.temp_path))
          pl.util.subprocess_capture_output(
            ['patch', root_dir, '--strip=0', '--unified', f'--input={tmp_patch_path}']
          )
        except Exception as exc:
          exception = ''.join(traceback_format_exception(exc))
          emess     = f'Application of patch based on {test_file} failed:\n{exception}\n'
          rej       = tmp_src.rej_file()
          if rej.exists():
            emess += f'\n{rej}:\n{rej.read_text()}'
          output_errors.append(emess)

    output_errors = [e for e in output_errors if e]
    if output_errors:
      pl.sync_print('\tNOT OK ', short_name)
      patch_error[test_file] = '\n'.join(output_errors)
    else:
      pl.sync_print('\tOK     ', short_name)
  if patch_error:
    err_str  = f"[ERROR] {85 * '-'} [ERROR]"
    err_bars = (err_str + '\n', err_str)
    for err_file in patch_error:
      pl.sync_print(patch_error[err_file].join(err_bars))
    return ReturnCode.ERROR_TEST_FAILED
  return ReturnCode.SUCCESS
