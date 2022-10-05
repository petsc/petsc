#!/usr/bin/env python3
#
# Usage:
#       Run gcov on the results of "make alltests" and create tar ball containing coverage results for one machine
#           ./gcov.py --run_gcov
#       Generate html pages showing coverage by merging tar balls from multiple machines (index_gcov1.html and index_gcov2.html)
#           ./gcov.py --merge_gcov  tarballs
#

from __future__ import print_function
import os
import glob
import inspect
import shutil
import operator
import optparse
import sys
import subprocess
from   time import gmtime,strftime
import tempfile
import pathlib

thisfile = os.path.abspath(inspect.getfile(inspect.currentframe()))
pdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(thisfile)))))
sys.path.insert(0, os.path.join(pdir, 'config'))

def subprocess_run(*args, **kwargs):
  if sys.version_info >= (3,7):
    output = subprocess.run(*args,**kwargs)
  else:
    if kwargs.pop("capture_output",None):
      kwargs.setdefault("stdout",subprocess.PIPE)
      kwargs.setdefault("stderr",subprocess.PIPE)
    output = subprocess.run(*args,**kwargs)
  return output

def pathlib_walk(path):
  dirs,files = [],[]
  for p in pathlib.Path(path).iterdir():
    if p.is_dir():
      dirs.append(p)
    else:
      files.append(p)
  # To emulate os.walk() we yield the current directory, and any subdirectories and files
  # in it. The caller can modify dirs to control which directories are decended into next
  yield path, dirs, files
  for d in dirs:
    yield from pathlib_walk(d)

def pathlib_append_suffix(path, suff):
  return pathlib.Path(str(path)+suff)

def pathlib_remove_all_suffix(path):
  # want to reduce 'foo.kokkos.cxx' to just 'foo'. Using a single .stem only results
  # in the outermost suffix being removed (i.e. 'foo.kokkos')
  path_stem = path.stem
  while 1:
    new_stem = pathlib.Path(path_stem).stem
    if new_stem == path_stem:
      break
    path_stem = new_stem
  return path_stem

def run_gcov(gcov_dir,petsc_dir,petsc_arch):

    # 1. Runs gcov
    # 2. Saves the tested source code line numbers and detected source code lines in
    #    xxx.c.tested and xxx.c.code files in gcov_dir

    print("Creating directory to save .tested and .code files\n")
    print("Running gcov\n")
    help = str(subprocess.check_output('gcov -h', shell=True).decode(encoding='UTF-8',errors='replace'))
    if help.find('--ignore-filename-regex') > -1: ignore_h = '--ignore-filename-regex="*.h" '
    else: ignore_h = ''
    print("gcov flags: %s" % ignore_h)

    # tuple of suffixes which the current installation can compile
    suffixes = tuple(set(
      subprocess_run(
        ['make','-f','gmakefile','print','VAR=langs'],
        capture_output=True, check=True, universal_newlines=True
      ).stdout.split()
    ))

    # no point in traversing into these directories
    skip_dirs = {'docs', 'binding'}
    petsc_dir = pathlib.Path(petsc_dir).resolve()
    obj_dir   = petsc_dir/petsc_arch/'obj'
    src_dir   = petsc_dir/'src'
    cwd_path  = pathlib.Path(os.getcwd())

    if cwd_path != petsc_dir:
      mess = '\n'.join([
        'Must be run from the same directory as source code was compiled from!',
        'This routine assumes that place is PETSC_DIR: {}'.format(petsc_dir),
        'Current working directory is:                 {}'.format(cwd_path)
      ])
      raise RuntimeError(mess)

    # walk the src tree search for files with allowed suffixes
    for par_dir, dirs, files in pathlib_walk(src_dir):
      dirs = [d for d in dirs if str(d) not in skip_dirs]
      for path in files:
        path_name = path.name
        if not path_name.endswith(suffixes):
          continue

        path_stem       = pathlib_remove_all_suffix(path)
        obj_path_base   = obj_dir/par_dir.relative_to(src_dir)/path_stem
        gcov_graph_file = obj_path_base.with_suffix('.gcno')
        gcov_data_file  = obj_path_base.with_suffix('.gcda')
        gcov_file       = pathlib_append_suffix(cwd_path/path_name, '.gcov')

        # no gcov data for this file? bail
        if not gcov_data_file.is_file() or not gcov_graph_file.is_file():
          continue

        # generate the source annotations
        cmd_list = ['gcov']
        if ignore_h:
          cmd_list.append(ignore_h)
        cmd_list.extend([
          '--object-directory={}'.format(gcov_data_file.parent),
          str(path)
        ])
        try:
          subprocess_run(cmd_list, capture_output=True, universal_newlines=True, check=True)
        except subprocess.CalledProcessError as cpe:
          print('*** subprocess error ***')
          print(cpe)
          print('stdout:')
          print(cpe.stdout)
          print('stderr:')
          print(cpe.stderr)

        if not gcov_file.exists():
          print('gcov output does not exist at expected path:', gcov_file)
          print('skipping',path)
          continue

        src           = path.read_text().splitlines()
        tested_output = []
        code_output   = []
        good_data     = True
        for line in gcov_file.read_text().splitlines():
          stripped = line.strip()
          if stripped.startswith('-:'):
            continue
          try:
            line_num = stripped.split(':')[1].strip()
          except IndexError as ie:
            print('Error processing', gcov_file, 'invalid gcov data, skipping data for file')
            print('  Line:', line)
            print('  Error message:')
            print(str(ie))
            good_data = False
            break

          assert line_num.isdigit(), 'Line number {} is not a number!'.format(line_num)

          if not stripped.startswith('#####'):
            tested_output.append(line_num)

          line_no = int(line_num)
          if line_no > 0 and 'SETERR' not in src[line_no-1]:
            code_output.append(line_num)

        if good_data:
          test_base = os.path.join(
            gcov_dir,'__'.join(('__'.join(par_dir.relative_to(petsc_dir).parts), path_name))
          )
          # create a 'tag' file, its existence indicates the file has been processed
          pathlib_append_suffix(test_base, '.tested').write_text('\n'.join(tested_output))
          # write the annotated version to file
          pathlib_append_suffix(test_base, '.code').write_text('\n'.join(code_output))
      for file in cwd_path.glob('*.gcov'):
        os.remove(file)

    print("""Finshed running gcov on PETSc source code""")
    return

def make_tarball(gcov_dir,petsc_dir,petsc_arch):

    # Create tarball of .lines files stored in gcov_dir
    print("""Creating tarball in %s to store gcov results files""" %(petsc_dir))
    curdir=os.path.abspath(os.path.curdir)
    os.chdir(gcov_dir)
    os.system("tar -czf "+petsc_dir+os.sep+"gcov.tar.gz *.tested *.code")
    os.chdir(petsc_dir)
    # Copy file so artifacts in CI propagate without overwriting
    shutil.copyfile('gcov.tar.gz',os.path.join(petsc_arch,'gcov.tar.gz'))
    print("""Tarball created in %s"""%(petsc_dir))
    os.chdir(curdir)
    return

def print_htmltable(nsrc_files,nsrc_files_not_tested,ntotal_lines,ntotal_lines_not_tested,output_list,out_fid,title,indx):
    if nsrc_files:
        print("""<h2><center>%s</center></h2>""" % title, file=out_fid)
        print("""<center><font size = "4">Number of testable source code files = %s</font></center>""" %(nsrc_files), file=out_fid)
        print("""<center><font size = "4">Number of testable source code files not tested fully = %s</font></center>""" %(nsrc_files_not_tested), file=out_fid)
        if float(nsrc_files) > 0: ratio = float(nsrc_files_not_tested)/float(nsrc_files)*100.0
        else: ratio = 0.0
        print("""<center><font size = "4">Percentage of testable source code files not tested fully = %3.2f</font></center><br>""" %ratio, file=out_fid)
        print("""<center><font size = "4">Total number of testable source code lines = %s</font></center>""" %(ntotal_lines), file=out_fid)
        print("""<center><font size = "4">Total number of testable source code lines not tested = %s</font></center>""" %(ntotal_lines_not_tested), file=out_fid)
        if float(ntotal_lines) > 0: ratio = float(ntotal_lines_not_tested)/float(ntotal_lines)*100.0
        else: ratio = 0.0
        print("""<center><font size = "4">Percentage of testable source code lines not tested = %3.2f</font></center>""" % ratio, file=out_fid)
    else:
        print("""<h2><center>%s</center></h2>""" % title, file=out_fid)
        print("""<center><font size = "4">No currently testable source code in branch was changed</font></center>""", file=out_fid)
    if output_list:
        print("""<center><font size = "4">%s</font></center>""" % indx, file=out_fid)
        print("""<table border="1" align = "center"><tr><th>Source code</th><th>Number of lines of testable source code</th><th>Number of lines not tested</th><th>% Code not tested</th></tr>""", file=out_fid)
        output_list.sort(key=operator.itemgetter(4),reverse=True)
        for l in output_list: # file_ctr in range(0,nsrc_files_not_tested):
          print("<tr><td><a href = %s>%s</a></td><td>%s</td><td>%s</td><td>%3.2f</td></tr>" % (l[1],l[0],l[2],l[3],l[4]), file=out_fid)
        print("""</table></p>""", file=out_fid)

def make_htmlpage(gcov_dir,petsc_dir,petsc_arch,tarballs,isCI,destBranch):
    # Create index_gcov webpages using information processed from running gcov

    cwd = os.getcwd()
    print("%s tarballs found\n%s" %(len(tarballs),tarballs))
    print("Extracting gcov directories from tar balls")
    #  Each tar file consists of a bunch of *.tested and *.code files NOT inside a directory
    tmp_dirs = []
    for i in range(0,len(tarballs)):
        dir = os.path.join(gcov_dir,str(i))
        os.mkdir(dir)
        os.system("cd "+dir+";gunzip -c "+tarballs[i] + "|tar -xof -")
        tmp_dirs.append(dir)

    # ---------------------- Stage 2 -----------------------------------
    print("Creating HTML files")
    temp_string = '<a name'
    spaces_12 = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp'
    sep = petsc_dir+os.sep+petsc_arch+os.sep+'obj'+os.sep

    date_time = strftime("%x %X %Z")
    outfile_name = petsc_dir+os.sep+petsc_arch+os.sep+'index_gcov.html'
    out_fid = open(outfile_name,'w')
    print("""<html><head><title>PETSc:Code Testing Statistics</title></head><body style="background-color: rgb(213, 234, 255);">""", file=out_fid)
    print("""<center>%s</center>"""%(date_time), file=out_fid)

    print("""<center><font size = "4"><a href = #fortran>Statistics for Fortran stubs</a></font></center>""", file=out_fid)
    # list of files that have been changed in the branch and introduce untested lines
    global_new_src_not_tested = []
    for lang in ['C','Fortran stubs']:

      if lang == 'Fortran stubs':print("""<a name = fortran></a>""", file=out_fid)
      print("Extracting data from files for %s" % lang)
      tested = {}  # for each line of each file has a 1 indicating it that line was tested
      code = {}  # for each line of each file has a 1 indicating if that line is source code (due to #ifdef different tarballs may have different source code lines)
      for j in tmp_dirs:
        if lang == 'C':
          files_dir1 = [i for i in os.listdir(j) if i.endswith('.tested') and not i.find('ftn-') > -1 and not i.find('f90-') > -1]
        if lang == 'Fortran stubs':
          files_dir1 = [i for i in os.listdir(j) if i.endswith('.tested') and (i.find('ftn-') > -1 or i.find('f90-') > -1)]

        for i in files_dir1:
            ii = i.replace('.tested','')
            in_file = os.path.join(j,i)
            in_fid = open(in_file,'r')
            testlines = in_fid.readlines()
            in_fid.close()
            if not ii in tested: tested[ii] = {}
            for line in testlines:
                try:
                  tested[ii][int(line)] = 1
                except Exception as e:
                  print("  Error processing %s" % in_file)
                  print("  Invalid tested data, skipping file")
                  print("  Error message %s" % str(e))
                  print("  Line:%s" % line)
            in_file = os.path.join(j,i.replace('tested','code'))
            in_fid = open(in_file,'r')
            codelines = in_fid.readlines()
            in_fid.close()
            if not ii in code: code[ii] = {}
            for line in codelines:
                code[ii][int(line)] = 1

      print("Building html files for %s" % lang)
      ntotal_lines = 0
      ntotal_lines_not_tested = 0
      output_list = []
      nsrc_files = 0
      nsrc_files_not_tested = 0
      for file in tested:
          nsrc_files += 1
          dir = os.path.dirname(petsc_dir+os.sep+petsc_arch+os.sep+'obj'+os.sep+file[5:].replace('__',os.sep))
          f = os.path.basename(file[5:].replace('__',os.sep))
          inhtml_file = os.path.join(dir,f+'.html')
          outhtml_file = os.path.join(dir,f+'.gcov.html')
          path = os.path.join(os.path.dirname('obj'+os.sep+file[5:].replace('__',os.sep)),f+'.gcov.html')
          try:
              inhtml_fid = open(inhtml_file,"r")
          except IOError:
              # Error check for files not opened correctly or file names not parsed correctly in stage 1
              raise RuntimeError("Cannot locate html file %s, run make srchtml first" % inhtml_file)
          lines = inhtml_fid.read().split('\n')
          inhtml_fid.close()

          temp_list = []
          temp_list.append(file.replace('__',os.sep))
          temp_list.append(path) # Relative path of hyperlink

          outhtml_fid = open(outhtml_file,"w")
          not_tested = 0
          n_code = 0
          for i in range(0,len(lines)):
              line = lines[i]
              if not i-9 in tested[file] and i-9 in code[file]:
                 if line.startswith('<pre width='):
                    num = line.find('>')
                    temp_outline = line[:num+1]+'<font color="red">Untested :&nbsp;&nbsp;</font>'+line[num+1:]
                 else:
                    temp_outline = '<font color="red">Untested :&nbsp;&nbsp;</font>'+line
                 not_tested += 1
              else:
                 temp_outline = spaces_12+line
              print(temp_outline,file = outhtml_fid)
              if i-9 in code[file]: n_code += 1
          outhtml_fid.close()
          nsrc_files_not_tested += (not_tested > 0)

          ntotal_lines += n_code
          ntotal_lines_not_tested += not_tested
          if n_code == 0:
             if not_tested > 0:
                 raise RuntimeError("Number source code lines is zero but number of untested lines is positive")
             else:
                 per_code_not_tested = 0
          else:
             per_code_not_tested = float(not_tested)/float(n_code)*100.0

          temp_list.append(n_code)
          temp_list.append(not_tested)
          temp_list.append(per_code_not_tested)
          output_list.append(temp_list)

      # Gather information on changes to source code and new source code
      new_nsrc_files = 0
      new_nsrc_files_not_tested = 0
      new_ntotal_lines = 0
      new_ntotal_lines_not_tested = 0
      new_output_list = []
      diff = str(subprocess.check_output('git diff --name-only '+destBranch+'...', shell=True).decode(encoding='UTF-8',errors='replace')).splitlines()
      ftn_keywords = ('ftn-', 'f90-')
      if lang == 'C':
         clang_suffixes = ('.c','.cxx','.cpp','.h','.hxx','.hpp','.cu','.cuh','.inl','.inc','.C','.cc')
         diff = [i for i in diff if i.endswith(clang_suffixes) and all(kw not in i for kw in ftn_keywords)]
      if lang == 'Fortran stubs':
         diff = [i for i in diff if i.endswith('.c') and any(kw in i for kw in ftn_keywords)]
      for file in diff:
         t_nsrc_lines = 0
         t_nsrc_lines_not_tested = 0
         ii = file.replace(os.sep,'__')
         try:
             diff = str(subprocess.check_output('git blame '+destBranch+'.. '+file+' | grep -v "^\^"', shell=True).decode(encoding='UTF-8',errors='replace')).split('\n')
         except:
             diff = ''
             pass
         lines_not_tested = {}
         for line in diff:
             if len(line) > 0:
                 line = line[:line.find(')')]
                 c = int(line[line.rfind(' '):])
                 if ii in code and c in code[ii]:
                     t_nsrc_lines += 1
                     if ii in tested and not c in tested[ii]:
                         t_nsrc_lines_not_tested += 1
                         lines_not_tested[c] = 1
         if t_nsrc_lines_not_tested:
            temp_list = []
            temp_list.append(file.replace('__',os.sep))

            dir = os.path.dirname(petsc_arch+os.sep+'obj'+os.sep+file[4:].replace('__',os.sep))
            f = os.path.basename(os.sep+file[4:].replace('__',os.sep))
            inhtml_file = os.path.join(dir,f+'.html')
            outshtml_file = os.path.join(dir,f+'.gcov_changed.html')
            path = os.path.join(os.path.dirname('obj'+os.sep+file[4:].replace('__',os.sep)),f+'.gcov_changed.html')
            temp_list.append(path) # Relative path of hyperlink
            outshtml_fid = open(outshtml_file,"w")
            try:
               inhtml_fid = open(inhtml_file,"r")
            except IOError:
               # Error check for files not opened correctly or file names not parsed correctly in stage 1
               raise RuntimeError("Cannot locate html file %s, run make srchtml first" % inhtml_file)
            lines = inhtml_fid.read().split('\n')
            inhtml_fid.close()
            for i in range(0,len(lines)):
               line = lines[i]
               if i-9 in lines_not_tested:
                 if line.startswith('<pre width='):
                    num = line.find('>')
                    temp_outline = line[:num+1]+'<font color="red">Untested :&nbsp;&nbsp;</font>'+line[num+1:]
                 else:
                    temp_outline = '<font color="red">Untested :&nbsp;&nbsp;</font>'+line
               else:
                 temp_outline = spaces_12+line
               print(temp_outline, file=outshtml_fid)
            outshtml_fid.close()
            new_nsrc_files += (t_nsrc_lines > 0)
            new_nsrc_files_not_tested += (t_nsrc_lines_not_tested > 0)
            new_ntotal_lines += t_nsrc_lines
            new_ntotal_lines_not_tested += t_nsrc_lines_not_tested
            temp_list.append(t_nsrc_lines)
            temp_list.append(t_nsrc_lines_not_tested)
            if t_nsrc_lines == 0:
                raise RuntimeError("Number source code lines is zero but number of untested lines is positive")
            else:
               per_code_not_tested = float(t_nsrc_lines_not_tested)/float(t_nsrc_lines)*100.0
            temp_list.append(per_code_not_tested)
            new_output_list.append(temp_list)

      if os.getenv('CI_COMMIT_BRANCH'):
          branchname = os.getenv('CI_COMMIT_BRANCH')
      else:
          branchname = str(subprocess.check_output('command git rev-parse --abbrev-ref HEAD', shell=True).decode(encoding='UTF-8',errors='replace'))
      global_new_src_not_tested.extend((fname, nlines) for fname,_,_,nlines,_ in new_output_list)
      print_htmltable(new_nsrc_files,new_nsrc_files_not_tested,new_ntotal_lines,new_ntotal_lines_not_tested,new_output_list,out_fid,'Changes in '+lang+' coverage data for branch '+branchname,'Lines marked with Untested are lines changed in the branch that are not tested')
      print_htmltable(nsrc_files,nsrc_files_not_tested,ntotal_lines,ntotal_lines_not_tested,output_list,out_fid,lang+' coverage data','Lines marked with Untested are any source code that has not been tested')
    print("""</body></html>""", file=out_fid)
    out_fid.close()

    print("Removing temporary directories created from tar files")
    for j in tmp_dirs:
        shutil.rmtree(j)

    print("End of gcov script")
    print("""See %s""" % os.path.join(petsc_dir,outfile_name))
    return global_new_src_not_tested

def main():

    parser = optparse.OptionParser(usage="%prog [options] ")
    parser.add_option('-p', '--petsc_dir', dest='petsc_dir',
                      help='Location of PETSC_DIR',
                      default='')
    parser.add_option('-a', '--petsc_arch', dest='petsc_arch',
                      help='Location of PETSC_ARCH',
                      default='')
    parser.add_option('-r', '--run_gcov', dest='run_gcov',
                      help='Running gcov and printing tarball',
                      action='store_true',default=False)
    parser.add_option('-m', '--merge_gcov', dest='merge_gcov',
                      help='Merging gcov results and creating main html page',
                      action='store_true',default=False)
    parser.add_option('-d', '--merge_branch', dest='merge_branch',
                      help='destination branch corresponding to the merge request',
                      default='')
    options, args = parser.parse_args()


    if options.petsc_dir:
        petsc_dir = options.petsc_dir
    else:
        petsc_dir = pdir
    if options.petsc_arch:
        petsc_arch = options.petsc_arch
    else:
        if 'PETSC_ARCH' in os.environ:
          if os.environ['PETSC_ARCH']:
            petsc_arch = os.environ['PETSC_ARCH']
        else:
            print("Must specify PETSC_ARCH with --petsc_arch")
            return

    if options.run_gcov:
        gcov_dir = tempfile.mkdtemp()
        print("Running gcov and creating tarball")
        run_gcov(gcov_dir,petsc_dir,petsc_arch)
        make_tarball(gcov_dir,petsc_dir,petsc_arch)
        shutil.rmtree(gcov_dir)
    elif options.merge_gcov:
        print("Creating main html page")
        tarballs = glob.glob(os.path.join(petsc_dir,'*.tar.gz'))

        # Gitlab CI organizes things differently
        isCI=False
        if len(tarballs)==0:
          tarballs=glob.glob(os.path.join(petsc_dir,'arch-*/gcov.tar.gz'))
          isCI=True

        if len(tarballs)==0:
          print("No coverage tarballs found")
          return

        print('options.merge_branch:',options.merge_branch)
        if options.merge_branch: destBranch = options.merge_branch
        else: destBranch = 'origin/main'
        print('destBranch:',destBranch)
        gcov_dir = tempfile.mkdtemp()
        untested_files = make_htmlpage(gcov_dir,petsc_dir,petsc_arch,tarballs,isCI,destBranch)
        shutil.rmtree(gcov_dir)
        if len(untested_files):
          print('*'*30,'WARNING','*'*30)
          print('Branch introduces new untested code!\n')
          for fname, linenos in untested_files:
            print('-',linenos,'lines in:',fname)
          sys.exit(1) # so CI fails
    else:
        parser.print_usage()

if __name__ == '__main__':
    main()

