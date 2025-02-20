#!/usr/bin/env python3
"""
# Created: Wed Oct  5 18:31:45 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import sys
import json
import shutil
import pathlib
import tempfile
import subprocess
import lxml
import lxml.etree
import copy
import functools
import textwrap
import os

# version of gcovr JSON format that this script was tested against and knows how to write
# see https://gcovr.com/en/stable/output/json.html#json-output
known_gcovr_json_versions = {'0.1', '0.2', '0.3', '0.5'}

class Logger:
  def __init__(self, *args, **kwargs):
    self._stdout = sys.stdout
    self._stderr = sys.stderr
    self.setup(*args, **kwargs)
    return

  def setup(self, stdout = None, stderr = None, verbosity = 0):
    if stdout is None:
      stdout = self._stdout

    if stderr is None:
      stderr = self._stderr

    self.flush(close=True)
    self.stdout  = stdout
    self.stderr  = stderr
    self.verbose = bool(verbosity)
    return

  @staticmethod
  def __log(stream, *args, **kwargs):
    kwargs.setdefault('file', stream)
    print(*args, **kwargs)
    return

  def log(self, *args, **kwargs):
    if self.verbose:
      self.__log(self.stdout, *args, **kwargs)
    return

  def log_error(self, *args, **kwargs):
    self.__log(self.stderr, *args, **kwargs)
    return

  def flush(self, close=False):
    for stream_name in ('stdout', 'stderr'):
      stream = getattr(self, stream_name, None)
      if stream:
        stream.flush()
        if close and stream not in {self._stdout, self._stderr}:
          stream.close()
    return

  def __del__(self):
    try:
      self.flush(close=True)
    except:
      pass
    return

gcov_logger = Logger()

@functools.total_ordering
class Version:
  def __init__(self, major, minor, micro):
    self.major     = int(major)
    self.minor     = int(minor)
    self.micro     = int(micro)
    self.__version = (self.major, self.minor, self.micro)
    return

  @classmethod
  def from_string(cls, ver):
    return cls.from_iterable(ver.split('.'))

  @classmethod
  def from_iterable(cls, it):
    version = list(map(int, it))
    assert len(version) <= 3
    while len(version) < 3:
      # pad out remaining minor, subminor versions
      version.append(0)
    return cls(*version)

  def __str__(self):
    return str(self.__version)

  def __repr__(self):
    return 'Version(major={}, minor={}, micro={})'.format(self.major, self.minor, self.micro)

  def __getitem__(self, idx):
    return self.__version[idx]

  def __eq__(self, other):
    if not isinstance(other, type(self)):
      other = self.from_iterable(other)
    return self.__version == other.__version

  def __lt__(self, other):
    if not isinstance(other, type(self)):
      other = self.from_iterable(other)
    return self.__version < other.__version

class GcovrRunner:
  def __init__(self, petsc_dir, verbosity):
    self.petsc_dir = petsc_dir
    self.verbosity = verbosity
    return

  @classmethod
  def gcovr_version(cls):
    attr_name = '__gcovr_version'
    version   = getattr(cls, attr_name, None)
    if version:
      return version

    raw_output = subprocess_check_output(['gcovr', '--version'])
    output     = raw_output.splitlines()[0].split()
    try:
      version = output[1]
    except IndexError as ie:
      mess = 'Invalid gcovr version string, cannot determine gcovr version from:\n{}'.format(raw_output)
      raise RuntimeError(mess) from ie

    version = Version.from_string(version)
    setattr(cls, attr_name, version)
    return version

  def build_command(self, *args):
    base_args = [
      'gcovr', '-j', '4', '--root', self.petsc_dir, '--exclude-throw-branches',
      '--exclude-unreachable-branches'
    ]
    if self.verbosity > 1:
      base_args.append('--verbose')

    args    = base_args + list(args)
    version = self.gcovr_version()
    if version < (5,) and '--html-self-contained' in args:
      # --html-self-contained since gcovr 5.0
      args.remove('--html-self-contained')
    if version < (5,1) and '--decisions' in args:
      # --decisions since gcovr 5.1
      args.remove('--decisions')
    if (5,) < version <= (5,2):
      # gcovr 5.1 and 5.2 add an assertion that merging functions (who have the same
      # mangled name) must have the same line number. Sounds sane in theory but does not
      # play well in practice, especially with macros. For example the following would
      # trigger an assertion:
      #
      # #if defined(FOO)
      #   int my_func() { return 0; }
      # #else
      #   int my_func() { retunr 1; }
      # #endif
      #
      # So to work around it we monkey-patch the gcovr executable every time to disable
      # the check
      monkey_patched = 'import sys; import gcovr; import gcovr.merging; gcovr.merging.DEFAULT_MERGE_OPTIONS.ignore_function_lineno = True; import gcovr.__main__; sys.exit(gcovr.__main__.main())'
      args = ['python3', '-c', monkey_patched] + args[1:]
    if version >= (6,):
      args.extend([
        '--exclude-noncode-lines',
        '--merge-mode-functions', 'separate',
      ])
    args.append('.')
    return list(map(str, args))

def sanitize_path(path):
  """
  Return a resolved pathlib.Path using PATH and assert it exists
  """
  path = pathlib.Path(path).resolve()
  assert path.exists(), 'path {} does not exist'.format(path)
  return path

def call_subprocess(func, *args, error_ok=False, **kwargs):
  gcov_logger.log('running', ' '.join(map(str, args[0])).join(("'","'")))
  ret = None
  try:
    ret = func(*args, **kwargs)
  except subprocess.CalledProcessError as cpe:
    return_code = cpe.returncode
    print('subprocess error, command returned exit code:', return_code)
    print('command:')
    print(' '.join(map(str, cpe.cmd)))
    print('stdout:')
    print(cpe.output)
    if getattr(cpe, 'stderr', None):
      print('stderr:')
      print(cpe.stderr)

    if error_ok:
      if isinstance(error_ok, int):
        error_ok = {error_ok}
      else:
        error_ok = set(map(int, error_ok))

      if return_code in error_ok:
        return ret
    raise cpe
  return ret

def subprocess_run(*args, **kwargs):
  if sys.version_info < (3,7):
    if kwargs.pop('capture_output', None):
      kwargs.setdefault('stdout', subprocess.PIPE)
      kwargs.setdefault('stderr', subprocess.PIPE)

  return call_subprocess(subprocess.run, *args, **kwargs)

def subprocess_check_output(*args, **kwargs):
  kwargs.setdefault('universal_newlines', True)
  return call_subprocess(subprocess.check_output, *args, **kwargs)

def load_report(json_file):
  """
  Read a json file JSON_FILE and yield each entry in files
  """
  with json_file.open() as fd:
    json_data = json.load(fd)

  try:
    json_format_version = json_data['gcovr/format_version']
  except KeyError:
    json_format_version = 'unknown'

  if str(json_format_version) not in known_gcovr_json_versions:
    mess = 'gcovr JSON version \'{}\' is incompatible, script is tested with version(s) {}'.format(
      json_format_version, known_gcovr_json_versions
    )
    raise RuntimeError(mess)

  return json_data

def store_report(json_data, dest_path):
  """
  Store JSON_DATA to disk at DEST_PATH
  """
  with dest_path.open('w') as fd:
    json.dump(json_data, fd)
  return dest_path

def get_branch_diff(merge_branch):
  """
  Get the diff between MERGE_BRANCH and current branch and return a dictionary RET which has entries:

  ret = {
    file_name : [list, of, lines, changed, by, branch],
  }
  """
  ret = {}

  merge_branch_name       = str(merge_branch)
  files_changed_by_branch = subprocess_check_output(
    ['git', 'diff', '--name-only', merge_branch_name + '...']
  ).splitlines()
  files_changed_by_branch = [f for f in files_changed_by_branch if not f.startswith('share/petsc/datafiles/') and os.path.basename(os.path.dirname(f)) != 'output']
  for file_name in files_changed_by_branch:
    blame_output = subprocess_run(
      ['git', 'blame', '-s', '--show-name', merge_branch_name + '..', file_name],
      capture_output=True, universal_newlines=True
    )

    try:
      blame_output.check_returncode()
    except subprocess.CalledProcessError:
      stderr = blame_output.stderr.strip()
      if stderr.startswith('fatal: no such path') and stderr.endswith('in HEAD'):
        # The branch removed a file from the repository. Since it no longer exists there
        # will obviously not be any coverage of it. So we ignore it.
        gcov_logger.log('File', "'"+file_name+"'", 'was deleted by branch. Skipping it')
        continue
      raise

    changed_ret = subprocess_run(
      ['grep', '-v', r'^\^'], input=blame_output.stdout, capture_output=True, universal_newlines=True
    )

    try:
      changed_ret.check_returncode()
    except subprocess.CalledProcessError:
      if changed_ret.returncode == 1:
        # git returns exitcode 1 if it finds nothing
        gcov_logger.log('File', "'"+file_name+"'", 'only contained deletions. Skipping it!')
        continue
      raise

    # each line in the blame is in the format
    #
    # commit_hash line_number) line_of_code
    #
    # we want a list of line_numbers
    ret[file_name] = [
      int(line[2].replace(')','')) for line in map(str.split, changed_ret.stdout.splitlines())
    ]
  return ret

def extract_tarballs(base_paths, dest_dir):
  """
  Search BASE_PATH for tarballs, and extract them to DEST_DIR
  """
  def unique_list(seq):
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]

  tar_files = []
  for path in base_paths:
    if path.is_dir():
      tar_files.extend(list(path.glob('*.json.tar.*')))
    else:
      assert path.is_file(), 'path {} is not a file'.format(path)
      tar_files.append(path)

  tar_files = unique_list(tar_files)
  if len(tar_files) == 0:
    mess = 'could not locate gcovr report tarballs in:\n{}'.format(
      '\n'.join(map(lambda p: '- '+str(p), base_paths))
    )
    raise RuntimeError(mess)

  gcov_logger.log('found', len(tar_files), 'tarball(s):')
  gcov_logger.log('- '+'\n- '.join(map(str, tar_files)))

  dest_dir.mkdir(exist_ok=True)
  for tarball in map(str, tar_files):
    gcov_logger.log('extracting', tarball, 'in directory', dest_dir)
    shutil.unpack_archive(tarball, extract_dir=str(dest_dir))
  return dest_dir

def merge_reports(runner, base_paths, dest_path):
  """
  Search BASE_PATH for a list of tarballs containing gcovr reports, unpack them and merge their
  contents. Write the merged result to DEST_PATH.
  """
  if dest_path.suffix != '.json':
    dest_path = pathlib.Path(str(dest_path) + '.json').resolve()

  try:
    dest_path.unlink()
  except FileNotFoundError:
    pass

  # unpack the tarballs in base_path and merge them if necessary
  with tempfile.TemporaryDirectory() as reports_path:
    reports_path = sanitize_path(reports_path)
    extract_tarballs(base_paths, reports_path)

    reports = [report for report in reports_path.iterdir() if report.name.endswith('.json')]
    assert len(reports) > 0, 'no gcovr reports in {}'.format(reports_path)

    gcov_logger.log('found', len(reports), 'report(s):')
    gcov_logger.log('- '+'\n- '.join(map(str, reports)))

    if len(reports) == 1:
      gcov_logger.log('copying', reports[0], 'to', dest_path)
      # only 1 report? no need to merge anything, just copy to new name
      return shutil.copy2(reports[0], dest_path)

    gcov_logger.log('merging reports to', dest_path)
    command = runner.build_command(
      '--json', '--output', dest_path, '--decisions', '--exclude-lines-by-pattern', r'^\s*SETERR.*'
    )
    for report in reports:
      command.extend(['--add-tracefile', report])
    gcov_logger.log(subprocess_check_output(command))

  return dest_path

def create_and_clear(dir_path, delete_pred = None):
  """
  Ensure directory at DIR_PATH exists (creating it if need be) and clear files in it according to
  DELETE_PRED. If DELETE_PRED is None, deletes all files in DIR_PATH. Not recursive.
  """
  if delete_pred is None:
      delete_pred = lambda p: p.is_file()

  if dir_path.exists():
    assert dir_path.is_dir(), "Directory path {} must be a directory".format(dir_path)
    for path in filter(delete_pred, dir_path.iterdir()):
      path.unlink()
  else:
    dir_path.mkdir()

  return dir_path

def generate_html(runner, merged_report, dest_dir, symlink_dir=None, report_name=None, html_title=None):
  """
  Generate a HTML coverage file from MERGED_REPORT in DEST_DIR. Optionally symlink the report base
  html file to SYMLINK_DIR. Optionally supply REPORT_NAME as the base name of the report, defaults to
  report.html. Optionally supply HTML_TITLE to set the title of the resulting report
  """
  if report_name is None:
    report_name = 'report.html'
  elif not report_name.endswith('.html'):
    report_name += '.html'

  if html_title is None:
    html_title = 'PETSc Code Coverage Report'

  html_title  = html_title.join(("'", "'"))
  dest_dir    = create_and_clear(dest_dir, delete_pred = lambda p: p.suffix.endswith('html'))
  report_path = dest_dir/report_name

  subprocess_check_output(
    runner.build_command(
      '--output', report_path,
      '--add-tracefile', merged_report,
      '--html-details',
      '--html-title', html_title,
      '--html-self-contained',
      '--sort-percentage',
      '--decisions',
      '--exclude-lines-by-pattern', r'^\s*SETERR.*',
      '--exclude', r'arch-ci.*'
    ),
    error_ok = 7 # return-code of 7 means some files were not found
  )

  symlink_name = None
  if symlink_dir is not None:
    assert symlink_dir.exists()
    symlink_name = symlink_dir/report_path.name
    try:
      symlink_name.unlink()
    except FileNotFoundError:
      pass
    symlink_name.symlink_to(report_path.relative_to(symlink_dir))
  return symlink_name

def generate_xml(runner, merged_report, dest_dir):
  """
  Generate a set of XML coverage files from MERGED_REPORT in DEST_DIR.
  """
  dest_dir    = create_and_clear(dest_dir, delete_pred = lambda p: p.suffix.endswith('xml'))
  mega_report = dest_dir/'mega_report.xml'

  ret = subprocess_check_output(
    runner.build_command(
      '--output', mega_report,
      '--add-tracefile', merged_report,
      '--xml-pretty',
      '--print-summary',
      '--exclude', r'arch-ci.*'
    ),
    error_ok = 7 # return-code of 7 means some files were not found
  )
  # print the output for CI
  print(ret)
  ## Workaround for https://gitlab.com/gitlab-org/gitlab/-/issues/328772. Pipeline
  ## artifacts are limited to 10M. So split the single cobertura xml (which is often
  ## >40MB) into one file per package, since there seems to be no limit on the _number_ of
  ## files just their size.
  orig_mega_xml_file = lxml.etree.fromstring(mega_report.read_bytes())

  # create a deep copy of the data, we want to preserve the metadata and structure of it,
  # but clear it of any "stuff". Note even though it is called 'empty_template' it is not
  # empty yet
  empty_template = copy.deepcopy(orig_mega_xml_file)
  packages       = empty_template.find('packages')

  # clear out all the existing packages in our copy of the data
  for p in packages:
    packages.remove(p)

  # 'empty_template' is now empty, i.e. contains only the header and description etc. Now
  # we go back through all the packages and use the template to create individual files
  # for each of the packages
  for package in orig_mega_xml_file.find('packages'):
    single_package_file = dest_dir/'report-{}.xml'.format(package.attrib['name'])
    gcov_logger.log("Creating package file {}".format(single_package_file))
    xml_to_write  = copy.deepcopy(empty_template)
    packages_node = xml_to_write.find('packages')

    # Add back the one package we want
    packages_node.append(package)

    single_package_file.write_bytes(lxml.etree.tostring(xml_to_write))

  # delete the mega report after we are done
  mega_report.unlink()
  return

def do_main(petsc_dir, petsc_arch, merge_branch, base_path, formats, verbosity, ci_mode):
  petsc_dir = sanitize_path(petsc_dir)
  assert petsc_dir.is_dir(), 'PETSC_DIR {} is not a directory'.format(petsc_dir)
  petsc_arch_dir = sanitize_path(petsc_dir/petsc_arch)
  base_path      = list(map(sanitize_path, base_path))
  if base_path[-1] != petsc_arch_dir:
    base_path.append(petsc_arch_dir)

  gcovr_dir = petsc_arch_dir/'gcovr'
  gcovr_dir.mkdir(exist_ok=True)

  if ci_mode:
    stdout_file = gcovr_dir/'merge_gcov.log'
    stderr_file = gcovr_dir/'merge_gcov_errors.log'
    # clear the files
    stdout_file.open('w').close()
    stderr_file.open('w').close()
    # reopen
    stdout = stdout_file.open('w')
    stderr = stderr_file.open('w')
  else:
    stdout = sys.stdout
    stderr = sys.stderr
  gcov_logger.setup(stdout, stderr, verbosity)

  runner        = GcovrRunner(petsc_dir, verbosity)
  merged_report = merge_reports(runner, base_path, gcovr_dir/'merged-gcovr-report.json')

  files_changed_by_branch = get_branch_diff(merge_branch)
  merged_report_json      = load_report(merged_report)

  total_testable_lines_by_branch = 0
  gcovr_report_version_str       = merged_report_json['gcovr/format_version']
  gcovr_report_version           = Version.from_string(gcovr_report_version_str)
  untested_code_by_branch        = {}
  untested_code_report           = {
    'gcovr/format_version' : gcovr_report_version_str,
    'files'                : []
  }

  if gcovr_report_version < (0, 5):
    line_exclusion = 'gcovr/noncode'
  elif gcovr_report_version == (0, 5):
    # Since JSON format version 0.5:
    # - The gcovr/noncode field was removed. Instead of generating noncode entries,
    #   the entire line is skipped.
    # - The gcovr/excluded field can be absent if false.
    line_exclusion = 'gcovr/excluded'
  else:
    # In addition to JSON format changes, also since gcovr 6.0:
    # - New --exclude-noncode-lines to exclude noncode lines. Noncode lines are not
    #   excluded by default anymore.
    #
    # should also check that empty lines are nicely handled.
    raise RuntimeError('Check that gcovr still handles report exclusions as above! See comment above')

  for data in merged_report_json['files']:
    file_name = data['file']
    if file_name not in files_changed_by_branch:
      continue

    changed_lines = set(files_changed_by_branch[file_name])
    cur_file_data = [
      line['line_number'] for line in data['lines']
      if line['line_number'] in changed_lines and line['count'] == 0 and not line.get(line_exclusion)
    ]

    if cur_file_data:
      # Make a copy of the line data, then iterate through and "invert" it, so that only
      # untested lines are left in. We achieve this by marking every line *except* new,
      # untested lines as "noncode". Gcovr ignores all noncode lines in the report.
      report_data = copy.deepcopy(data)
      for line in report_data['lines']:
        if line['line_number'] in changed_lines and line['count'] == 0:
          # only ignore untested lines added by the branch
          continue
        if gcovr_report_version < (0, 5):
          line['gcovr/noncode'] = True
        else:
          line['gcovr/excluded'] = True

      untested_code_report['files'].append(report_data)
      untested_code_by_branch[file_name] = cur_file_data

    total_testable_lines_by_branch += len(changed_lines)
    # a minor performance optimization, we remove the processed file from the list of
    # files to check for, and if we don't have any more to check for we can just bail
    files_changed_by_branch.pop(file_name)
    if len(files_changed_by_branch.keys()) == 0:
      break

  # generate the html report
  if 'html' in formats:
    # CI mode unconditionally creates the untested line report even if there are no
    # untested lines since the environment must have a valid file to load...
    if ci_mode or untested_code_report['files']:
      untested_report = store_report(untested_code_report, gcovr_dir/'untested-gcovr-report.json')
      generate_html(
        runner, untested_report, gcovr_dir/'html_untested',
        symlink_dir=gcovr_dir, report_name='report_untested.html',
        html_title='PETSc Untested Code Report'
      )
    generate_html(runner, merged_report, gcovr_dir/'html', symlink_dir=gcovr_dir)

  if 'xml' in formats:
    generate_xml(runner, merged_report, gcovr_dir/'xml')

  ret_code     = 0
  ci_fail_file = gcovr_dir/'.CI_FAIL'
  try:
    ci_fail_file.unlink()
  except FileNotFoundError:
    pass
  if untested_code_by_branch:
    def num_uncovered_lines_allowed(num_lines_changed):
      import math

      if num_lines_changed < 10:
        # small MRs must cover all changed code
        return 0
      return math.floor(num_lines_changed / (7.0 * math.log(num_lines_changed)))

    ret_code    = 1
    warn_banner = ' WARNING '.join(('*'*40, '*'*40))
    mini_bar    = '-'*5
    gcov_logger.log_error(warn_banner)
    gcov_logger.log_error('This branch introduces untested new code!')
    gcov_logger.log_error('')
    gcov_logger.log_error(mini_bar, 'summary:')
    # print a summary first
    for file_name, lines in untested_code_by_branch.items():
      gcov_logger.log_error('-', len(lines), 'line(s) in', file_name)
    gcov_logger.log_error('')
    gcov_logger.log_error(mini_bar, 'detailed breakdown:')
    for file_name, lines in untested_code_by_branch.items():
      gcov_logger.log_error('\n-', '{}:'.format(file_name))
      with open(file_name) as fd:
        src_lines = fd.readlines()
      for line in lines:
        gcov_logger.log_error('{}:'.format(line), src_lines[line - 1], end='')
    gcov_logger.log_error(warn_banner)
    gcov_logger.log_error('NOTE:')
    gcov_logger.log_error('\n'.join((
      '- If you believe this is a false positive (covered code accused of being uncovered), check again! The vast majority of packages *are* run through coverage.',
      '',
      '- If code is of the form:',
      '',
      '    if (condition) {',
      '      SETERRQ(...); <--- line marked as untested',
      '    }',
      '',
      '  Use PetscCheck()/PetscAssert() instead, they will properly count the error line as tested',
      '',
      '- If the code is part of an extended (multi-line) error path, it is better to explicitly test such code as described at https://petsc.org/main/developers/testing/#testing-errors-and-exceptional-code'
    )))

    # flush stdout, pythons print is line buffered and since we don't end with newline in
    # the prints above it may not have flushed
    gcov_logger.flush()

    # gobble the error code if we are in CI mode. The CI job must not fail, otherwise the
    # environments hosting the reports are not deployed. Instead we signal the error by
    # creating a special .CI_FAIL file in the arches gcovr directory
    if ci_mode:
      ret_code           = 0
      num_untested_lines = sum(map(len, untested_code_by_branch.values()))
      if num_untested_lines > num_uncovered_lines_allowed(total_testable_lines_by_branch):
        # have more uncovered code than was allowed, the CI pipeline must ultimately fail
        ci_fail_file.touch()

  return ret_code

def make_error_exc():
  def add_logfile_path(mess, stream_name):
    try:
      path = getattr(gcov_logger, stream_name)
    except AttributeError:
      path = 'unknown location'
    mess.append('  {}: {}'.format(stream_name, path))
    return mess

  width = 90
  bars  = '=' * width
  mess  = textwrap.wrap('An error occurred while processing GCOVR results. NOTE THAT DEBUG LOGS ARE LOCATED:', width=width-2, initial_indent='  ', subsequent_indent='  ')
  add_logfile_path(mess, 'stdout')
  add_logfile_path(mess, 'stderr')
  mess.insert(0, bars)
  mess.append(bars)
  return Exception('\n' + '\n'.join(mess))

def main(*args, **kwargs):
  try:
    return do_main(*args, **kwargs)
  except Exception as e:
    try:
      exc = make_error_exc()
    except Exception as mem:
      exc = mem
    raise exc from e

if __name__ == '__main__':
  import argparse

  petsc_dir  = os.environ.get('PETSC_DIR')
  petsc_arch = os.environ.get('PETSC_ARCH')

  parser = argparse.ArgumentParser('PETSc gcovr utility')
  parser.add_argument('--petsc_dir', default=petsc_dir, required=petsc_dir is None, type=pathlib.Path, help='PETSc directory')
  parser.add_argument('--petsc_arch', default=petsc_arch, required=petsc_arch is None, help='PETSc build directory name')
  parser.add_argument('-b', '--merge-branch', help='destination branch corresponding to the merge request')
  parser.add_argument('-c', '--ci-mode', action='store_true', help='enable CI mode, which adds all arch-ci-* folders in PETSC_DIR to the base search path, and overrides the log output files')
  parser.add_argument('-p','--base-path', type=pathlib.Path, nargs='*', help='base path containing tarball of gcovr report files for analysis, may be repeated to add multiple base paths')
  parser.add_argument('--html', action='store_true', help='generate HTML output')
  parser.add_argument('--xml', action='store_true', help='generate XML output')
  parser.add_argument('-v', '--verbose', action='count', default=0, help='verbose output, multiple flags increases verbosity')
  parser.add_argument('-l', '--log-output-stdout', default='stdout', const='stdout', nargs='?', help='Output file (or file stream) to log informational output to')
  parser.add_argument('-e', '--log-output-stderr', default='stderr', const='stderr', nargs='?', help='Output file (or file stream) to log errors to')
  args = parser.parse_args()

  formats = [attr for attr in ('html', 'xml') if getattr(args, attr)]

  if len(formats) == 0:
    parser.error('Must supply one of --html or --xml or both')

  for stream_name in ('stdout', 'stderr'):
    attr = 'log_output_' + stream_name
    if getattr(args, attr) == stream_name:
      setattr(args, attr, getattr(sys, stream_name))

  gcov_logger.setup(args.log_output_stdout, args.log_output_stderr, args.verbose)

  args.petsc_dir = sanitize_path(args.petsc_dir)

  if args.base_path is None:
    args.base_path = [args.petsc_dir]

  if args.ci_mode:
    args.base_path.extend(list(args.petsc_dir.glob('arch-*')))

  if not args.merge_branch:
    args.merge_branch = subprocess_check_output(
      [args.petsc_dir/'lib'/'petsc'/'bin'/'maint'/'check-merge-branch.sh']
    ).strip()

  sys.exit(
    main(
      args.petsc_dir, args.petsc_arch, args.merge_branch, args.base_path, formats,
      args.verbose, args.ci_mode
    )
  )
