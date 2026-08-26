#!/usr/bin/env python3
"""
Post review findings on a GitLab merge request as inline DiffNote comments,
for the /review-mr-post skill in .agents/skills/.

  ai_review_post.py IID FINDINGS.json [--meta FILE] [--dry-run]

FINDINGS.json holds a list of objects, each with

  file   path of the changed file on the new side of the merge request diff,
         that is the b/ path of its header, even where the file is renamed
  line   line number in the new version of that file
  body   comment text, optionally ending in a ```suggestion block

The commit shas the comments are anchored to, and the pre-rename path of every
file the merge request renames, are read from mr-IID-meta.json in the
repository root, written there by ai_review_fetch.py.  FINDINGS.json and a
--meta FILE resolve against the invocation directory, as usual.  One POSTED
or FAILED line is printed per finding, followed by a POSTED_OK and
POSTED_FAILED count.  A --dry-run posts nothing, counts the findings it
checked as DRY_RUN instead of POSTED_OK, and flags each suggestion block that
would be demoted to a plain code block.

Exit status: 0 every finding posted, or a --dry-run that reached the end, 1
otherwise.
"""
import os
import re
import sys
import json
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_review_fetch import DEFAULT_TIMEOUT, EXIT_FAIL, EXIT_OK, IID_RE, chdir_root, die, emit, flatten  # noqa: E402

SUGGESTION_RE = re.compile(r'(```suggestion[^\n]*\n)(.*?)(```)\s*$', re.DOTALL)
OPEN_RE       = re.compile(r'(?m)^```suggestion')
FENCE_RE      = re.compile(r'(?m)^ {0,3}`{3,}')

def guard_suggestion(body):
  """Demote a suggestion block GitLab would render incorrectly to a plain code block."""
  # The block's opening fence is the last line-start ```suggestion preceded by
  # an even number of fences: an odd count means the candidate sits inside an
  # open block, and an inline mention never opens one.  Anchoring there leaves
  # an earlier, closed quoted block alone while still catching a ```suggestion
  # nested inside the trailing block.
  starts = [m.start() for m in OPEN_RE.finditer(body) if len(FENCE_RE.findall(body, 0, m.start())) % 2 == 0]
  if not starts: return body
  match = SUGGESTION_RE.match(body, starts[-1])
  if not match: return body
  inner = match.group(2)
  if inner.count('```') or inner.rstrip('\n').endswith('\\'): return body[:match.start(1)] + '```\n' + body[match.end(1):]
  return body

def load_findings(path):
  try:
    with open(path) as fd: findings = json.load(fd)
  except (OSError, ValueError) as exc:
    die('cannot read findings from %s: %s' % (path, exc))
  if not isinstance(findings, list): die('%s must hold a list of findings' % path)
  for index, finding in enumerate(findings):
    if not isinstance(finding, dict): die('finding %d in %s is not an object' % (index, path))
    for key in ('file', 'line', 'body'):
      if key not in finding: die('finding %d in %s has no "%s"' % (index, path, key))
    if not isinstance(finding['line'], int) or isinstance(finding['line'], bool):
      die('finding %d in %s has a non-integer line %r' % (index, path, finding['line']))
    if finding['line'] < 1: die('finding %d in %s has a non-positive line %d' % (index, path, finding['line']))
    if not isinstance(finding['file'], str) or not finding['file'].strip(): die('finding %d in %s has an empty or non-string file' % (index, path))
    if not isinstance(finding['body'], str) or not finding['body'].strip(): die('finding %d in %s has an empty or non-string body' % (index, path))
  return findings

def load_meta(path):
  """Return the shas comments are anchored to and the new to old path map of the renamed files."""
  try:
    with open(path) as fd: meta = json.load(fd)
  except (OSError, ValueError) as exc:
    die('cannot read merge request metadata from %s: %s' % (path, exc))
  refs    = meta.get('diff_refs') or {}
  missing = [key for key in ('base_sha', 'head_sha', 'start_sha') if not refs.get(key)]
  if missing: die('%s is missing diff_refs.%s' % (path, ', diff_refs.'.join(missing)))
  return refs, meta.get('renames') or {}

def post(iid, refs, renames, finding, body, timeout):
  """Create one inline DiffNote and return (discussion id, None), or (None, error message) on failure."""
  payload = {
    'body': body,
    'position': {
      'position_type': 'text',
      'base_sha': refs['base_sha'],
      'head_sha': refs['head_sha'],
      'start_sha': refs['start_sha'],
      'new_path': finding['file'],
      'old_path': renames.get(finding['file'], finding['file']),
      'new_line': finding['line'],
    },
  }
  cmd = ['glab', 'api', 'projects/:id/merge_requests/%s/discussions' % iid,
         '-X', 'POST', '--input', '-', '-H', 'Content-Type: application/json']
  try:
    proc = subprocess.run(cmd, input=json.dumps(payload).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
  except FileNotFoundError:
    die('glab is not in PATH')
  except subprocess.TimeoutExpired:
    return None, 'glab timed out after %d seconds' % timeout
  # glab puts GitLab's error body on stdout and only a terse note on stderr; keep
  # both, and never return an empty message for a failure.
  if proc.returncode: return None, flatten(proc.stderr + b' ' + proc.stdout) or 'glab exited %d with no output' % proc.returncode
  try:
    discussion = json.loads(proc.stdout)
  except ValueError:
    return None, 'glab did not return JSON: %s' % proc.stdout[:200].decode('utf-8', 'replace')
  notes = discussion.get('notes') or []
  if not notes or notes[0].get('type') != 'DiffNote':
    return None, 'GitLab created a %s, not an inline DiffNote' % (notes[0].get('type') if notes else 'discussion with no note')
  return discussion.get('id'), None

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('iid', help='merge request IID')
  # abspath at parse time, before chdir_root, so the paths resolve against the invocation directory.
  parser.add_argument('findings', metavar='FINDINGS.json', type=os.path.abspath, help='file holding the findings to post')
  parser.add_argument('--meta', metavar='FILE', type=os.path.abspath, help='merge request metadata (default: mr-IID-meta.json)')
  parser.add_argument('--dry-run', action='store_true', help='validate and print the findings without posting')
  parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, metavar='SECONDS',
                      help='limit on each glab invocation (default: %d)' % DEFAULT_TIMEOUT)
  args = parser.parse_args()

  if not IID_RE.match(args.iid): die('invalid merge request IID %r' % args.iid)
  chdir_root(args.timeout)
  findings      = load_findings(args.findings)
  refs, renames = load_meta(args.meta or 'mr-%s-meta.json' % args.iid)

  posted = 0
  failed = 0
  for finding in findings:
    where = '%s:%d' % (finding['file'], finding['line'])
    body  = guard_suggestion(finding['body'])
    if args.dry_run:
      print('DRY-RUN %s%s' % (where, '' if body == finding['body'] else ' (the suggestion block will be demoted to a plain code block)'))
      continue
    discussion, error = post(args.iid, refs, renames, finding, body, args.timeout)
    if error is not None:
      print('FAILED %s %s' % (where, error))
      failed += 1
    else:
      print('POSTED %s %s' % (where, discussion))
      posted += 1
  if args.dry_run:
    emit('DRY_RUN', len(findings))
  else:
    emit('POSTED_OK', posted)
    emit('POSTED_FAILED', failed)
  return EXIT_FAIL if failed else EXIT_OK

if __name__ == '__main__':
  sys.exit(main())
