#!/usr/bin/env python3
#
#  Provides a PETSc search engine capability using Tantivy
#
##
import os
import re
import sys
import pathlib
import subprocess
import tantivy

def addSemicolon(line):
  '''Adds semicolon to end of synopsis function prototype lines in PETSc md manual pages'''
  if line.startswith('PetscErrorCode '): return line + ';'
  else: return line

# horribly crude but all the available tools are useless
def ProcessMarkDown(text):
  '''Remove cruft from markdown files'''

  footer = text.find('[Index of all')
  if footer != -1:
    text = text[:footer]
  #indices = 'Index of all ' + mansec + ' routines Table of Contents for all manual pages Index of all manual pages'
  indices = ' routines Table of Contents for all manual pages Index of all manual pages'
  bibs    = '{bibliography} :filter: docname in docnames'
  text = text.replace('/edit/', '/blob/')
  text = text.replace('Edit on GitLab','Source file location on GitLab')
  text = text.replace(indices,'').replace('()','').replace(bibs,'').replace(':orphan:','')
  text = re.subn(':any:[a-zA-Z0-9_]*','',text)[0]
  text = text.replace('## References\n```{bibliography}\n:filter: docname in docnames\n```\n','')
  text = text[0:text.find('[Index of all')]
  # RecursiveCharacterTextSplitter.split_text() messes up ``` unless they are completely separated from neighboring lines
  # text =  text.replace('```','\n```\n')
  # doctext does not end generated #include with ; at end
  text = '\n'.join([addSemicolon(s) for s in text.split('\n')])
  # langchain rules cannot handle RST .. formatting
  for r in ['.. ', ':maxdepth:', ':']:
    split = text.split('\n')
    text = '\n'.join([s for s in split if not s.startswith(r)])
  for r in ['{toctree}', '{literalinclude}', 'PETSC_DOC_OUT_ROOT_PLACEHOLDER']:
    split = text.split('\n')
    text = '\n'.join([s for s in split if not s.find(r) > -1])
  text = text.replace('**','')
  text = text.replace('`console\n','')
  text = text.replace('`text\n','')
  text = text.replace('`','')
  text = text.replace('###','')
  text = text.replace('##','')
  text = text.replace('#','')
  text = re.sub(r'\([a-zA-Z0-9_]*\)=\n','',text)
  return text
#
def createDocsSchema():
  schema_builder = tantivy.SchemaBuilder()
  schema_builder.add_text_field("file", stored=True, tokenizer_name='raw')
  schema_builder.add_text_field("path", stored=True, tokenizer_name='raw')
  schema_builder.add_text_field("body", stored=True, tokenizer_name='en_stem')
  schema = schema_builder.build()
  path = os.path.join(os.environ['PETSC_DIR'], os.environ['PETSC_ARCH'], 'tantivy', 'index', 'docs')
  return (schema, path)

def generateDocsIndex():
  import shutil

  (schema, path) = createDocsSchema()
  if os.path.isdir(path): shutil.rmtree(path)
  if not os.path.isdir(path): os.makedirs(path)
  index = tantivy.Index(schema, path = path)

  # could maybe use TextAnalyzerBuilder()

  writer = index.writer()
  for root, _, files in os.walk(os.path.join(os.environ['PETSC_DIR'], os.environ['PETSC_ARCH'] + '-doc')):
    if '_build/html/manual' in root: continue
    if 'changes' in root: continue
    for file_name in files:
      if file_name == 'singleindex.md': continue
      if file_name == 'index.md': continue
      file_path = os.path.join(root, file_name)
      if not file_path.endswith('.md'): continue
      if file_path.endswith('RegisterAll.md'): continue
      if file_path.endswith('Register.md'): continue
      if file_path.endswith('Package.md'): continue
      with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
      content = ProcessMarkDown(content)
      document = tantivy.Document()
      document.add_text("file", file_name)
      document.add_text("path", file_path)
      document.add_text("body", content)
      writer.add_document(document)
  writer.commit()
  index.reload()

def searchDocsIndex(text: str, cnt: int = 10, md: bool = False):
  """Find the files with the most appropriate matches to text. md = True indicates return the MarkDown file, otherwise return the HTML file"""
  (schema, path) = createDocsSchema()
  if not os.path.isdir(path): raise RuntimeError("Use lib/petsc/bin/search.py --generate to create the search index")
  index = tantivy.Index(schema, path = path)
  searcher = index.searcher()

  query = index.parse_query(text, ["body"])
  top = searcher.search(query, cnt).hits
  files = []
  for i in range(0,min(cnt,len(top))):
    file = searcher.doc(top[i][1])['path'][0]
    if not md:
      if file.find('manualpages') > -1:
        file = file.replace('manualpages/','_build/html/manualpages/').replace('.md','.html')
      else:
        file = file.replace(os.environ['PETSC_ARCH'] + '-doc/',os.environ['PETSC_ARCH'] + '-doc/_build/html/').replace('.md','.html')
    files.append(file)
  return files

if __name__ ==  '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '--generate':
    generateDocsIndex()
  else:
    import webbrowser

    if len(sys.argv) > 2 and sys.argv[1] == '-n':
      cnt = int(sys.argv[2])
      argv = sys.argv[3:]
    else:
      cnt = 10
      argv = sys.argv[1:]

    browser = False
    if len(argv) > 1 and argv[0] == '--md':
      files = searchDocsIndex(' '.join(sys.argv[2:]), cnt = cnt, md = True)
    elif len(argv) > 1 and argv[0] == '--browser':
      files = searchDocsIndex(' '.join(argv[1:]), cnt = cnt)
      browser = True
    else:
      files = searchDocsIndex(' '.join(argv[0:]), cnt = cnt)
    for f in files:
      if browser: webbrowser.open_new_tab('file://' + f)
      print(f)