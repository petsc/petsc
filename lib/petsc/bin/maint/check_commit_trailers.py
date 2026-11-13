#!/usr/bin/env python3
"""Check that commit messages credit LLM assistance with Assisted-by:, not Co-authored-by:."""

import argparse
import re
import subprocess
import sys

# Match an LLM tool's email address or name. A bare "Claude" needs a model word to match.
LLM = re.compile(r'''\b(?:noreply@anthropic|codex@openai|cursoragent@cursor)\.com
                   | \bclaude[\s:._-]*(?:code|opus|sonnet|haiku|fable)\b
                   | \b(?:codex|chatgpt|copilot|gemini|llm)\b''', re.IGNORECASE | re.VERBOSE)

# Trailer value -> whether to flag it. --self-check tests these samples.
SAMPLES = {
  'Claude Opus 5 (1M context) <noreply@anthropic.com>': True,
  'Claude Code <noreply@anthropic.com>': True,
  'Claude:claude-opus-5[1m]': True,
  'ChatGPT Astra 6 <codex@openai.com>': True,
  'Cursor <cursoragent@cursor.com>': True,
  'Satish Balay <balay@mcs.anl.gov>': False,
  'Claude Lefebvre <claude.lefebvre@example.fr>': False,
  'Jane Doe <jane.doe@openai.com>': False,
}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('range', nargs='?', metavar='BASE..HEAD', help='revision range to check')
  parser.add_argument('--self-check', action='store_true', help='verify the patterns against known trailer values and exit')
  args = parser.parse_args()

  if args.self_check:
    for value, flagged in sorted(SAMPLES.items()):
      if bool(LLM.search(value)) != flagged: sys.exit('misclassified: %s' % value)
    print('%d trailer values classified correctly' % len(SAMPLES))
    return
  if not args.range: parser.error('a revision range is required')

  # Let git parse the trailer block, so that a Co-authored-by: written in prose is not a trailer.
  log = subprocess.run(['git', 'log', '--reverse', '--no-merges', args.range,
                        '--format=%x1e%h %s%n%(trailers:key=Co-authored-by,valueonly,separator=%x1f)'],
                       stdout=subprocess.PIPE, check=True, universal_newlines=True).stdout
  errors = []
  for entry in log.split('\x1e')[1:]:
    header, _, trailers = entry.partition('\n')
    for value in trailers.strip().split('\x1f'):
      if LLM.search(value): errors.append('%s\n      Co-authored-by: %s' % (header.strip(), value))

  if errors:
    sys.exit('Credit an LLM with "Assisted-by:", not "Co-authored-by:", in:\n    %s\n'
             'Amend the commit messages, for example with git rebase -i.' % '\n    '.join(errors))


if __name__ == '__main__':
  sys.exit(main())
