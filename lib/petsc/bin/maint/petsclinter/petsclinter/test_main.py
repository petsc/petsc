#!/usr/bin/env python3
"""
# Created: Tue Jun 21 09:25:37 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import shutil
import difflib
import tempfile
import traceback
import petsclinter as pl

from .classes import Path

class TemporaryCopy:
  __slots__ = 'fname', 'tmp', 'temp_path'

  def __init__(self, fname):
    self.fname = Path(fname).resolve(strict=True)
    return

  def __enter__(self):
    self.tmp       = tempfile.NamedTemporaryFile(suffix=self.fname.suffix)
    self.temp_path = Path(self.tmp.name).resolve(strict=True)
    shutil.copy2(str(self.fname), str(self.temp_path))
    return self

  def __exit__(self, *args, **kwargs):
    Path.unlink(self.orig_file(), missing_ok=True)
    Path.unlink(self.rej_file(), missing_ok=True)
    del self.tmp
    return

  def orig_file(self):
    return self.temp_path.append_suffix('.orig')

  def rej_file(self):
    return self.temp_path.append_suffix('.rej')

def test_main(petsc_dir, test_path, output_dir, patches, errors_fixed, errors_left, replace=False, verbose=False):
  def test(generated_output, reference_file):
    short_ref_name = reference_file.relative_to(petsc_dir)
    if replace:
      pl.sync_print('\tREPLACE', short_ref_name)
      reference_file.write_text(''.join(generated_output))
      return
    if not reference_file.exists():
      return f'Missing reference file \'{reference_file}\'\n'
    return ''.join(
      difflib.unified_diff(
        reference_file.read_text().splitlines(True), generated_output,
        fromfile=str(short_ref_name), tofile='Generated Output', n=0
      )
    )

  # sanitize the output so that it will be equal across systems
  def sanitize_output_file(text):
    return [] if text is None else [l.replace(str(petsc_dir), '.') for l in text.splitlines(True)]

  def sanitize_patch_file(text):
    # skip the diff header with file names
    return [] if text is None else text.splitlines(True)[2:]

  def rename_patch_file_target(text, new_path):
    lines    = text.splitlines(True)
    out_file = lines[0].split()[1]
    lines[0] = lines[0].replace(out_file, str(new_path))
    lines[1] = lines[1].replace(out_file, str(new_path))
    return ''.join(lines)


  patch_error = {}
  root_dir    = f'--directory={petsc_dir.anchor}'
  patches     = dict(patches)
  output      = {p : ['<--- FIXED --->', s ,'<--- LEFT --->'] for p, s in errors_fixed}
  for path, string in errors_left:
    if path not in output:
      output[path] = ['<--- FIXED --->\n<--- LEFT --->']
    output[path].append(string)
  output = {key : '\n'.join(val if len(val) == 4 else val + ['']) for key, val in output.items()}
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
          tmp_patch_path.write_text(rename_patch_file_target(patches.get(test_file), tmp_src.temp_path))
          pl.util.subprocess_run(
            ['patch', root_dir, '--strip=0', '--unified', f'--input={tmp_patch_path}'],
            check=True, universal_newlines=True, capture_output=True
          )
        except Exception as re:
          exception = ''.join(traceback.format_exception(re, chain=True))
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
    err_bars = f"[ERROR] {85 * '-'} [ERROR]"
    err_bars = (err_bars + '\n', err_bars)
    for err_file in patch_error:
      pl.sync_print(patch_error[err_file].join(err_bars))
    return 21
  return 0
