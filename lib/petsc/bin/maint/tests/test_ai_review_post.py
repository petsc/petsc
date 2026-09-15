"""Exercise reconciliation through the CLI entry point without running glab or using the network."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

import pytest


SPEC = importlib.util.spec_from_file_location('ai_review_post', Path(__file__).parents[1] / 'ai_review_post.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)

REFS = {'base_sha': 'a' * 40, 'head_sha': 'b' * 40, 'start_sha': 'c' * 40}
FINDING = {'file': 'src/sys/example.c', 'line': 12, 'body': 'Check this return value.'}


def response(data, status=200, headers=None, code=None):
  # glab prints the status and headers with LF, then a CRLF separator.
  head = 'HTTP/2.0 %d Test response\nContent-Type: application/json\n' % status
  head += ''.join('%s: %s\n' % item for item in (headers or {}).items())
  raw = data if isinstance(data, bytes) else json.dumps(data).encode()
  return subprocess.CompletedProcess([], int(status >= 400) if code is None else code,
                                     head.encode() + b'\r\n' + raw, b'')


def discussion(finding=FINDING, head=REFS['head_sha'], identifier='existing', inline=True, commit_id=None):
  note = {'id': 1, 'body': finding['body'], 'type': 'DiffNote' if inline else 'DiscussionNote',
          'commit_id': commit_id, 'system': False}
  if inline:
    note['position'] = dict(REFS, head_sha=head, new_path=finding['file'], old_path=finding['file'],
                            new_line=finding['line'], position_type='text')
  return {'id': identifier, 'notes': [note]}


class GitLab:
  def __init__(self):
    self.discussions = []
    self.posts = []
    self.pages = []
    self.events = []
    self.actions = []
    self.page_results = {}

  def run(self, cmd, *, input, stdout, stderr, timeout):
    assert cmd[:2] == ['glab', 'api']
    assert '--include' in cmd
    assert stdout == subprocess.PIPE and stderr == subprocess.PIPE
    assert timeout == review.DEFAULT_TIMEOUT
    if input is None:
      assert '-X' not in cmd
      query = parse_qs(urlsplit(cmd[2]).query)
      page = int(query['page'][0])
      per_page = int(query['per_page'][0])
      self.pages.append(page)
      self.events.append(('GET', page))
      if page in self.page_results:
        result = self.page_results[page]
        if isinstance(result, Exception): raise result
        return result
      end = page * per_page
      headers = {'X-Page': str(page), 'X-Next-Page': str(page + 1) if end < len(self.discussions) else '',
                 'X-Total': str(len(self.discussions)), 'X-Per-Page': str(per_page)}
      return response(self.discussions[end - per_page:end], headers=headers)
    assert cmd[cmd.index('-X') + 1] == 'POST'
    assert cmd[cmd.index('--input') + 1] == '-'
    assert cmd[cmd.index('-H') + 1] == 'Content-Type: application/json'
    payload = json.loads(input)
    assert 'commit_id' not in payload
    self.posts.append(payload)
    self.events.append(('POST', payload['body']))
    action = self.actions.pop(0) if self.actions else 'accept'
    if isinstance(action, Exception): raise action
    if isinstance(action, subprocess.CompletedProcess): return action
    accepted = discussion({'body': payload['body'].replace('\r', '').rstrip(), 'file': payload['position']['new_path'],
                           'line': payload['position']['new_line']}, head=payload['position']['head_sha'],
                          identifier='posted-%d' % len(self.posts), inline=action != 'noninline')
    if action != 'noninline': accepted['notes'][0]['position'] = copy.deepcopy(payload['position'])
    self.discussions.append(accepted)
    if action == 'timeout_after_accept': raise subprocess.TimeoutExpired(cmd, timeout)
    if action == 'bad_json_after_accept': return response(b'{incomplete', status=201)
    return response(accepted, status=201)


@pytest.fixture
def gitlab(monkeypatch):
  api = GitLab()
  monkeypatch.setattr(review.subprocess, 'run', api.run)
  monkeypatch.setattr(review, 'chdir_root', lambda timeout: None)
  return api


@pytest.fixture
def run_review(tmp_path, monkeypatch, capsys, gitlab):
  def run(findings=None, *, renames=None, dry_run=False):
    findings_path, meta_path = tmp_path / 'findings.json', tmp_path / 'meta.json'
    findings_path.write_text(json.dumps([FINDING] if findings is None else findings))
    meta_path.write_text(json.dumps({'diff_refs': REFS, 'renames': renames or {}}))
    argv = ['ai_review_post.py', '17', str(findings_path), '--meta', str(meta_path)]
    if dry_run: argv.append('--dry-run')
    monkeypatch.setattr(sys, 'argv', argv)
    code = review.main()
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err
  return run


def test_already_posted_finding(gitlab, run_review):
  gitlab.discussions = [discussion()]
  code, lines, _ = run_review()
  assert code == 0 and not gitlab.posts
  assert 'PRESENT src/sys/example.c:12 existing' in lines
  assert 'POSTED_OK=0' in lines and 'POSTED_PRESENT=1' in lines


@pytest.mark.parametrize('change', [
  {'head_sha': 'd' * 40}, {'new_path': 'src/other.c'}, {'new_line': 13},
])
def test_same_body_at_another_revision_or_location_is_posted(change, gitlab, run_review):
  previous = discussion()
  previous['notes'][0]['position'].update(change)
  gitlab.discussions = [previous]
  code, lines, _ = run_review()
  assert code == 0 and len(gitlab.posts) == 1
  assert 'POSTED_OK=1' in lines and 'POSTED_PRESENT=0' in lines


def test_renamed_path_uses_new_side_for_matching_and_both_paths_for_posting(gitlab, run_review):
  old_path = 'src/sys/old.c'
  previous = discussion()
  previous['notes'][0]['position']['old_path'] = old_path
  gitlab.discussions = [previous]
  new_finding = dict(FINDING, line=13)
  code, lines, _ = run_review([FINDING, new_finding], renames={FINDING['file']: old_path})
  assert code == 0 and len(gitlab.posts) == 1
  assert lines[0].startswith('PRESENT ')
  assert gitlab.posts[0]['position'] == dict(REFS, position_type='text', old_path=old_path,
                                           new_path=FINDING['file'], new_line=13)
  assert 'commit_id' not in gitlab.posts[0]


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_guarded_body_matches_and_is_used_for_new_posts(newline, gitlab, run_review):
  finding = dict(FINDING, body='Fix this.\n```suggestion\n#define FOO \\\n```')
  finding['body'] = finding['body'].replace('\n', newline)
  guarded = 'Fix this.\n```\n#define FOO \\\n```'
  gitlab.discussions = [discussion(dict(finding, body=guarded))]
  code, lines, _ = run_review([finding, dict(finding, line=13)])
  assert code == 0 and lines[0].startswith('PRESENT ')
  assert len(gitlab.posts) == 1 and gitlab.posts[0]['body'] == guarded


@pytest.mark.parametrize('ending', ['\n', ' \t\n', '\r\n'])
def test_normalized_body_is_posted_and_recognized_on_rerun(ending, gitlab, run_review):
  body = 'Check this return value.\n```suggestion\n  PetscCall(Foo());\n```'
  findings = [dict(FINDING, body=body + ending), dict(FINDING, line=13)]
  code, lines, _ = run_review(findings)
  assert code == 0 and len(gitlab.posts) == 2
  assert gitlab.posts[0]['body'] == body
  assert 'POSTED_OK=2' in lines and 'POSTED_UNCERTAIN=0' in lines
  code, lines, _ = run_review(findings)
  assert code == 0 and len(gitlab.posts) == 2
  assert 'POSTED_PRESENT=2' in lines and 'POSTED_OK=0' in lines


@pytest.mark.parametrize('ending', ['\n', ' \t\n', '\r\n'])
def test_existing_body_with_whitespace_is_recognized(ending, gitlab, run_review):
  gitlab.discussions = [discussion(dict(FINDING, body=FINDING['body'] + ending))]
  code, lines, _ = run_review()
  assert code == 0 and not gitlab.posts
  assert lines[0].startswith('PRESENT ')


@pytest.mark.parametrize('body', ['Check this return value.\r\n', 'Fix.\n```suggestion\nNew line\n```\n'])
def test_normalization_is_not_reported_as_suggestion_demotion(body, gitlab, run_review):
  code, lines, _ = run_review([dict(FINDING, body=body)], dry_run=True)
  assert code == 0 and not gitlab.posts
  assert lines[0] == 'DRY-RUN src/sys/example.c:12'


def test_all_pages_are_read_before_any_post(gitlab, run_review):
  gitlab.discussions = [discussion(dict(FINDING, body='Unrelated %d' % i), identifier=str(i)) for i in range(100)]
  gitlab.discussions.append(discussion())
  code, lines, _ = run_review([FINDING, dict(FINDING, line=13)])
  assert code == 0 and gitlab.pages == [1, 2]
  assert gitlab.events[:2] == [('GET', 1), ('GET', 2)]
  assert len(gitlab.posts) == 1 and lines[0].startswith('PRESENT ')


def test_partial_success_then_rerun_same_file(gitlab, run_review):
  findings = [FINDING, dict(FINDING, line=13)]
  gitlab.actions = ['accept', response({'message': 'invalid position'}, status=422)]
  code, lines, _ = run_review(findings)
  assert code == 1
  assert 'POSTED_OK=1' in lines and 'POSTED_FAILED=1' in lines and 'POSTED_UNCERTAIN=0' in lines
  assert any(line.startswith('FAILED src/sys/example.c:13 HTTP 422') for line in lines)
  code, lines, _ = run_review(findings)
  assert code == 0 and len(gitlab.discussions) == 2 and len(gitlab.posts) == 3
  assert lines[0].startswith('PRESENT ') and lines[1].startswith('POSTED ')
  assert 'POSTED_PRESENT=1' in lines and 'POSTED_OK=1' in lines


@pytest.mark.parametrize('action', ['timeout_after_accept', 'bad_json_after_accept'])
def test_uncertain_acceptance_then_deliberate_rerun(action, gitlab, run_review):
  findings = [FINDING, dict(FINDING, line=13)]
  gitlab.actions = [action]
  code, lines, _ = run_review(findings)
  assert code == 1 and len(gitlab.posts) == 1
  assert lines[0].startswith('UNCERTAIN ')
  assert lines[1].startswith('FAILED src/sys/example.c:13 not attempted')
  assert 'POSTED_UNCERTAIN=1' in lines and 'POSTED_FAILED=2' in lines
  code, lines, _ = run_review(findings)
  assert code == 0 and len(gitlab.posts) == 2 and len(gitlab.discussions) == 2
  assert lines[0].startswith('PRESENT ') and lines[1].startswith('POSTED ')


def test_duplicate_findings_in_one_file_are_not_reposted(gitlab, run_review):
  code, lines, _ = run_review([FINDING, FINDING])
  assert code == 0 and len(gitlab.posts) == 1
  assert lines[1].startswith('PRESENT ')


def test_noninline_response_stops_posting_and_blocks_rerun(gitlab, run_review):
  gitlab.actions = ['noninline']
  code, lines, _ = run_review([FINDING, dict(FINDING, line=13)])
  assert code == 1 and len(gitlab.posts) == 1
  assert lines[0].startswith('UNCERTAIN ')
  assert lines[1].startswith('FAILED src/sys/example.c:13 not attempted')
  assert 'POSTED_UNCERTAIN=1' in lines and 'POSTED_FAILED=2' in lines
  assert 'POSTED_PRESENT=0' in lines and 'POSTED_OK=0' in lines
  code, lines, _ = run_review()
  assert code == 1 and len(gitlab.posts) == 1
  assert lines[0].startswith('FAILED ') and 'has no revision' in lines[0]


def test_noninline_note_on_reviewed_revision_is_present(gitlab, run_review):
  gitlab.discussions = [discussion(inline=False, commit_id=REFS['head_sha'])]
  code, lines, _ = run_review()
  assert code == 0 and not gitlab.posts
  assert lines[0].startswith('PRESENT_NONINLINE ')


def test_noninline_note_on_other_revision_does_not_match(gitlab, run_review):
  gitlab.discussions = [discussion(inline=False, commit_id='d' * 40)]
  code, lines, _ = run_review()
  assert code == 0 and len(gitlab.posts) == 1 and lines[0].startswith('POSTED ')


def test_legacy_noninline_note_without_revision_blocks_guessing(gitlab, run_review):
  gitlab.discussions = [discussion(inline=False)]
  code, lines, _ = run_review()
  assert code == 1 and not gitlab.posts
  assert lines[0].startswith('FAILED ') and 'has no revision' in lines[0]


def test_matching_reply_is_found(gitlab, run_review):
  thread = discussion(dict(FINDING, body='A different root note'))
  thread['notes'].append(discussion()['notes'][0])
  gitlab.discussions = [thread]
  code, lines, _ = run_review()
  assert code == 0 and not gitlab.posts and lines[0].startswith('PRESENT ')


def test_dry_run_reports_present_and_absent_findings_without_posting(gitlab, run_review):
  gitlab.discussions = [discussion(), discussion(dict(FINDING, body='Not inline'), inline=False,
                                               identifier='noninline', commit_id=REFS['head_sha'])]
  demoted = dict(FINDING, body='Fix.\n```suggestion\n#define FOO \\\n```')
  code, lines, _ = run_review([FINDING, dict(FINDING, body='Not inline'), demoted], dry_run=True)
  assert code == 0 and not gitlab.posts
  assert lines[0].startswith('PRESENT ') and lines[1].startswith('PRESENT_NONINLINE ')
  assert lines[2].startswith('DRY-RUN ') and 'demoted' in lines[2]
  assert 'DRY_RUN=1' in lines and 'POSTED_PRESENT=2' in lines and 'POSTED_OK=0' in lines


@pytest.mark.parametrize('result', [
  response({'message': 'forbidden'}, status=403),
  subprocess.TimeoutExpired(['glab'], review.DEFAULT_TIMEOUT),
  FileNotFoundError('glab is not in PATH'),
  response(b'{truncated', headers={'X-Page': '1', 'X-Next-Page': ''}),
  response({'message': 'not a list'}, headers={'X-Page': '1', 'X-Next-Page': ''}),
  response([{'id': 'missing-notes'}], headers={'X-Page': '1', 'X-Next-Page': ''}),
  response([{'id': 'bad-note', 'notes': [None]}], headers={'X-Page': '1', 'X-Next-Page': ''}),
  response([], headers={'X-Page': '1'}),
  response([], headers={'X-Page': '2', 'X-Next-Page': ''}),
  response([], headers={'X-Page': '1', 'X-Next-Page': '3'}),
  response([], headers={'X-Page': '1', 'X-Next-Page': '2'}),
  response([], headers={'X-Page': '1', 'X-Next-Page': '', 'X-Total': '1'}),
  response([], headers={'X-Page': '1', 'X-Next-Page': '', 'X-Total-Pages': '2'}),
  response([], headers={'X-Page': '1', 'X-Next-Page': '', 'X-Total-Pages': '-1'}),
])
@pytest.mark.parametrize('dry_run', [False, True])
def test_failed_or_incomplete_listing_aborts_without_posting(result, dry_run, gitlab, run_review):
  gitlab.page_results[1] = result
  code, lines, err = run_review(dry_run=dry_run)
  assert code == 1 and not gitlab.posts
  assert 'cannot list all MR discussions; nothing posted' in err
  assert lines[0].startswith('FAILED ')
  assert 'POSTED_OK=0' in lines and 'POSTED_FAILED=1' in lines


def test_failed_second_page_prevents_even_first_page_posts(gitlab, run_review):
  gitlab.page_results = {
    1: response([discussion()], headers={'X-Page': '1', 'X-Next-Page': '2'}),
    2: response({'message': 'server error'}, status=500),
  }
  code, lines, _ = run_review([dict(FINDING, line=13)])
  assert code == 1 and gitlab.pages == [1, 2] and not gitlab.posts
  assert 'POSTED_FAILED=1' in lines


def test_repeated_page_is_rejected(gitlab, run_review):
  gitlab.page_results = {
    1: response([discussion()], headers={'X-Page': '1', 'X-Next-Page': '2'}),
    2: response([discussion()], headers={'X-Page': '2', 'X-Next-Page': ''}),
  }
  code, _, err = run_review()
  assert code == 1 and not gitlab.posts and 'pagination repeated' in err


def test_pagination_without_optional_totals(gitlab, run_review):
  gitlab.page_results = {
    1: response([discussion(dict(FINDING, body='Another note'), identifier='other')],
                headers={'X-Page': '1', 'X-Next-Page': '2'}),
    2: response([discussion()], headers={'X-Page': '2', 'X-Next-Page': ''}),
  }
  code, lines, _ = run_review()
  assert code == 0 and gitlab.pages == [1, 2] and not gitlab.posts
  assert lines[0].startswith('PRESENT ')


def test_changed_total_during_pagination_is_rejected(gitlab, run_review):
  gitlab.page_results = {
    1: response([discussion()], headers={'X-Page': '1', 'X-Next-Page': '2', 'X-Total': '2'}),
    2: response([discussion(identifier='second')], headers={'X-Page': '2', 'X-Next-Page': '', 'X-Total': '3'}),
  }
  code, _, err = run_review()
  assert code == 1 and not gitlab.posts and 'total changed' in err


@pytest.mark.parametrize('result', [
  response({'message': 'server error'}, status=500),
  response({'message': 'request timeout'}, status=408),
  subprocess.CompletedProcess([], 1, b'', b'connection reset'),
  response(None, status=201),
  response([], status=201),
  response({'id': 'no-notes'}, status=201),
  response(discussion(head='d' * 40), status=201),
  response(discussion(dict(FINDING, line=13)), status=201),
  OSError('pipe read failed after sending the request'),
])
def test_unusable_post_responses_are_uncertain(result, gitlab, run_review):
  gitlab.actions = [result]
  code, lines, _ = run_review()
  assert code == 1 and len(gitlab.posts) == 1
  assert lines[0].startswith('UNCERTAIN ')
  assert 'POSTED_FAILED=1' in lines and 'POSTED_UNCERTAIN=1' in lines


def test_missing_glab_at_post_time_is_a_known_failure(gitlab, run_review):
  gitlab.actions = [FileNotFoundError('glab is not in PATH')]
  code, lines, _ = run_review()
  assert code == 1 and not gitlab.discussions
  assert lines[0].startswith('FAILED ') and 'could not start glab' in lines[0]
  assert 'POSTED_UNCERTAIN=0' in lines
