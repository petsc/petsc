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
--meta FILE resolve against the invocation directory, as usual. Before posting,
every discussion page is fetched and checked. Matching notes are skipped using
the normalized, guarded body, new path, new line, and head SHA. Normalization
removes carriage returns and trailing whitespace to match GitLab. Non-inline
notes match by body and commit_id; a matching non-inline body without a
revision blocks that finding because its origin cannot be established.

Each finding prints POSTED, PRESENT, PRESENT_NONINLINE, FAILED, or UNCERTAIN.
UNCERTAIN means a POST may have been accepted but its response is unusable; no
further posts are attempted in that run. Stop and report it before considering
a rerun. Otherwise, rerun the same findings file to reconcile partial success.
POSTED_OK counts new inline notes, POSTED_PRESENT counts both PRESENT statuses
(including newly accepted non-inline notes), and POSTED_UNCERTAIN counts unknown
outcomes. For compatibility, POSTED_FAILED counts all FAILED and UNCERTAIN
findings. A --dry-run posts nothing, prints DRY-RUN for absent findings, flags
demoted suggestions, and adds a DRY_RUN count. It still reconciles existing
notes and reports failures.

Exit status: 0 if every finding is posted, present, or would be posted in a
dry run; 1 if reconciliation fails or any finding is FAILED or UNCERTAIN.
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

def discussion_request(iid, timeout, page=1, payload=None):
  """Capture headers as well as JSON so pagination and HTTP failures are explicit."""
  path = 'projects/:id/merge_requests/%s/discussions' % iid
  if payload is None: path += '?per_page=100&page=%d' % page
  cmd = ['glab', 'api', path, '--include']
  if payload is not None: cmd += ['-X', 'POST', '--input', '-', '-H', 'Content-Type: application/json']
  return subprocess.run(cmd, input=None if payload is None else json.dumps(payload).encode(),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

def response_parts(raw):
  """Parse the HTTP status, headers, and body emitted by glab api --include."""
  parts = re.split(rb'\r?\n\r?\n', raw, maxsplit=1)
  if len(parts) != 2: raise ValueError('glab did not return complete HTTP headers')
  lines = parts[0].decode('utf-8', 'replace').splitlines()
  match = re.fullmatch(r'HTTP/[0-9.]+ (\d{3})(?: .*)?', lines[0]) if lines else None
  if not match: raise ValueError('glab did not return an HTTP status')
  headers = {}
  for line in lines[1:]:
    key, separator, value = line.partition(':')
    if not separator: raise ValueError('glab returned a malformed HTTP header')
    headers[key.lower()] = value.strip()
  return int(match.group(1)), headers, parts[1]

def discussion_notes(discussion):
  """Reject incomplete discussion objects instead of silently losing match candidates."""
  if not isinstance(discussion, dict) or not isinstance(discussion.get('id'), str) or not discussion['id']:
    raise ValueError('discussion has no valid id')
  notes = discussion.get('notes')
  if not isinstance(notes, list) or not notes: raise ValueError('discussion %s has no notes' % discussion['id'])
  for note in notes:
    if not isinstance(note, dict) or not note.get('id') or not isinstance(note.get('body'), str):
      raise ValueError('discussion %s has an incomplete note' % discussion['id'])
    if note.get('position') is not None and not isinstance(note['position'], dict):
      raise ValueError('discussion %s has an invalid position' % discussion['id'])
  return notes

def load_discussions(iid, timeout):
  """Read all pages before allowing any POST; missing pagination evidence is fatal."""
  discussions, seen = [], set()
  page, total = 1, None
  while True:
    proc = discussion_request(iid, timeout, page=page)
    if proc.returncode:
      raise ValueError('discussion page %d: %s' % (page, flatten(proc.stderr + b' ' + proc.stdout) or 'glab failed'))
    status, headers, raw = response_parts(proc.stdout)
    if status != 200: raise ValueError('discussion page %d returned HTTP %d' % (page, status))
    if headers.get('x-page') != str(page) or 'x-next-page' not in headers:
      raise ValueError('discussion page %d lacks consistent X-Page/X-Next-Page headers; completeness is unknown' % page)
    next_page = headers['x-next-page']
    if next_page not in ('', str(page + 1)): raise ValueError('discussion pagination skips or repeats a page')
    if 'x-total-pages' in headers:
      pages = int(headers['x-total-pages'])
      if pages < 0 or page > max(1, pages) or bool(next_page) != (page < pages):
        raise ValueError('discussion pagination contradicts X-Total-Pages')
    if 'x-total' in headers:
      count = int(headers['x-total'])
      if count < 0 or (total is not None and total != count): raise ValueError('discussion total changed during pagination')
      total = count
    batch = json.loads(raw)
    if not isinstance(batch, list): raise ValueError('discussion page %d is not a JSON array' % page)
    if next_page and not batch: raise ValueError('discussion page %d is empty before the last page' % page)
    for discussion in batch:
      discussion_notes(discussion)
      if discussion['id'] in seen: raise ValueError('discussion pagination repeated %s' % discussion['id'])
      seen.add(discussion['id'])
      discussions.append(discussion)
    if not next_page:
      if total is not None and len(discussions) != total: raise ValueError('discussion listing is incomplete: expected %d, received %d' % (total, len(discussions)))
      return discussions
    page += 1

def find_present(discussions, refs, finding, body):
  """Return a matching note's status and discussion id, or a blocking ambiguity."""
  noninline, ambiguous = None, None
  for discussion in discussions:
    for note in discussion['notes']:
      if note.get('system') or note['body'].replace('\r', '').rstrip() != body: continue
      position = note.get('position') or {}
      if position:
        if (position.get('new_path'), position.get('new_line')) != (finding['file'], finding['line']): continue
        if position.get('head_sha') == refs['head_sha']: return 'PRESENT', discussion['id']
        if not position.get('head_sha'): ambiguous = discussion['id']
      elif note.get('commit_id') == refs['head_sha']:
        noninline = discussion['id']
      elif not note.get('commit_id'):
        ambiguous = discussion['id']
  if noninline: return 'PRESENT_NONINLINE', noninline
  if ambiguous: return 'FAILED', 'matching note in discussion %s has no revision; inspect it before retrying' % ambiguous
  return None, None

def post(iid, refs, renames, finding, body, timeout):
  """Return (status, detail, accepted discussion), distinguishing rejection from uncertainty."""
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
  try:
    proc = discussion_request(iid, timeout, payload=payload)
  except FileNotFoundError as exc:
    return 'FAILED', 'could not start glab: %s' % exc, None
  except subprocess.TimeoutExpired:
    return 'UNCERTAIN', 'glab timed out after %d seconds; GitLab may have accepted the request' % timeout, None
  except OSError as exc:
    return 'UNCERTAIN', 'glab communication failed; GitLab may have accepted the request: %s' % exc, None
  try:
    status, _, raw = response_parts(proc.stdout)
  except ValueError as exc:
    return 'UNCERTAIN', '%s: %s' % (exc, flatten(proc.stderr + b' ' + proc.stdout)), None
  if 400 <= status < 500 and status != 408:
    return 'FAILED', 'HTTP %d: %s' % (status, flatten(proc.stderr + b' ' + raw)), None
  if proc.returncode or status != 201:
    return 'UNCERTAIN', 'HTTP %d, glab exit %d: %s' % (status, proc.returncode, flatten(proc.stderr + b' ' + raw)), None
  try:
    discussion = json.loads(raw)
    notes = discussion_notes(discussion)
    present, _ = find_present([discussion], refs, finding, body)
  except ValueError as exc:
    return 'UNCERTAIN', 'unusable creation response: %s' % exc, None
  if present == 'PRESENT' and notes[0].get('type') == 'DiffNote':
    return 'POSTED', discussion['id'], discussion
  if present == 'PRESENT_NONINLINE':
    return present, discussion['id'], discussion
  return 'UNCERTAIN', 'creation response does not confirm the requested inline note and revision', None

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

  counts = dict.fromkeys(('POSTED', 'PRESENT', 'PRESENT_NONINLINE', 'FAILED', 'UNCERTAIN', 'DRY_RUN'), 0)
  blocked = None
  try:
    discussions = load_discussions(args.iid, args.timeout)
  except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
    discussions = []
    blocked = 'cannot list all MR discussions; nothing posted: %s' % exc
    print(blocked, file=sys.stderr)
  for finding in findings:
    where = '%s:%d' % (finding['file'], finding['line'])
    normalized_body = finding['body'].replace('\r', '').rstrip()
    body = guard_suggestion(normalized_body)
    status, detail = find_present(discussions, refs, finding, body)
    if status is None and blocked: status, detail = 'FAILED', blocked
    if status is None and args.dry_run:
      print('DRY-RUN %s%s' % (where, '' if body == normalized_body else ' (the suggestion block will be demoted to a plain code block)'))
      counts['DRY_RUN'] += 1
      continue
    if status is None:
      status, detail, discussion = post(args.iid, refs, renames, finding, body, args.timeout)
      if discussion is not None: discussions.append(discussion)
      if status == 'UNCERTAIN': blocked = 'not attempted after an UNCERTAIN result; stop and report it before retrying'
    print('%s %s %s' % (status, where, detail))
    counts[status] += 1
  if args.dry_run: emit('DRY_RUN', counts['DRY_RUN'])
  emit('POSTED_OK', counts['POSTED'])
  emit('POSTED_FAILED', counts['FAILED'] + counts['UNCERTAIN'])
  emit('POSTED_PRESENT', counts['PRESENT'] + counts['PRESENT_NONINLINE'])
  emit('POSTED_UNCERTAIN', counts['UNCERTAIN'])
  return EXIT_FAIL if blocked or counts['FAILED'] or counts['UNCERTAIN'] else EXIT_OK

if __name__ == '__main__':
  sys.exit(main())
