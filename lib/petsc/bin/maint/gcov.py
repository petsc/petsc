#!/usr/bin/python
#
# Usage:
#       Run gcov on the results of "make alltests" and create tar ball containing coverage results for one machine
#           ./gcov.py --run_gcov
#       Generate html pages showing coverage by merging tar balls from multiple machines (index_gcov1.html and index_gcov2.html)
#           ./gcov.py --merge_gcov [LOC] tarballs
#

from __future__ import print_function
import os
import glob
import inspect
import shutil
import operator
import optparse
import sys
from   time import gmtime,strftime

thisfile = os.path.abspath(inspect.getfile(inspect.currentframe()))
pdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(thisfile)))))
sys.path.insert(0, os.path.join(pdir, 'config'))

def run_gcov(gcov_dir,petsc_dir,petsc_arch):

    # 1. Runs gcov
    # 2. Saves the untested source code line numbers in
    #    xxx.c.lines files in gcov_dir

    print("Creating directory to save .lines files\n")
    if os.path.isdir(gcov_dir):
        shutil.rmtree(gcov_dir)
    os.mkdir(gcov_dir)
    print("Running gcov\n")
    for root,dirs,files in os.walk(os.path.join(petsc_dir,"src")):
        # Directories to skip
        if (root.find('tests') != -1) | (root.find('tutorials') != -1) | (root.find('benchmarks') != -1)| (root.find('examples') != -1) | (root.find('src'+os.sep+'dm'+os.sep+'mesh') != -1) | (root.find('draw'+os.sep+'impls'+os.sep+'win32') != -1) | (root.find('impls'+os.sep+'python') != -1) :
            continue
        os.chdir(root)
        for file_name in files:
            csrc_file = file_name.endswith('.c')
            if csrc_file:
                c_file = file_name.split('.c')[0]
                OBJDIR = os.path.join(petsc_dir, petsc_arch, 'obj')
                objpath = os.path.join(OBJDIR, os.path.relpath(c_file, os.path.join(petsc_dir,"src")))
                gcov_graph_file = objpath+".gcno"
                gcov_data_file  = objpath+".gcda"
                if os.path.isfile(gcov_graph_file) and os.path.isfile(gcov_data_file):
                    # gcov created .gcno and .gcda files => create .gcov file,parse it and save the untested code line
                    # numbers in .lines file
                    os.system('gcov --object-directory "%s" "%s"' % (os.path.dirname(gcov_data_file), file_name))
                    gcov_file = file_name+".gcov"
                    try:
                        gcov_fid = open(gcov_file,'r')
                        root_tmp1 = root.split(petsc_dir+os.sep)[1].replace(os.sep,'_')
                        lines_fid = open(os.path.join(gcov_dir,root_tmp1+'_'+file_name+'.lines'),'w')
                        for line in gcov_fid:
                            if line.find("#####") > 0:
                                line_num = line.split(":")[1].strip()
                                print("""%s"""%(line_num), file=lines_fid)
                        gcov_fid.close()
                        lines_fid.close()
                    except IOError:
                        continue
                else:
                    # gcov did not create .gcno or .gcda file,save the source code line numbers to .lines file
                    file_id = open(file_name,'r')
                    root_tmp1 = root.split(petsc_dir+os.sep)[1].replace(os.sep,'_')
                    lines_fid = open(os.path.join(gcov_dir,root_tmp1+'_'+file_name+'.lines'),'w')
                    nlines = 0
                    line_num = 1
                    in_comment = 0
                    for line in file_id:
                        if line.strip() == '':
                            line_num += 1
                        else:
                            if line.lstrip().startswith('/*'):
                                in_comment = 1
                            if in_comment == 0:
                                print("""%s"""%(line_num), file=lines_fid)
                            if in_comment & (line.find('*/') != -1):
                                in_comment = 0
                            line_num += 1
                    file_id.close()
                    lines_fid.close()
    print("""Finshed running gcov on PETSc source code""")
    return

def make_tarball(dirname,petsc_dir,petsc_arch):

    # Create tarball of .lines files stored in gcov_dir
    print("""Creating tarball in %s to store gcov results files""" %(petsc_dir))
    curdir=os.path.abspath(os.path.curdir)
    os.chdir(dirname)
    os.system("tar -czf "+petsc_dir+os.sep+"gcov.tar.gz *.lines")
    os.chdir(petsc_dir)
    shutil.rmtree(dirname)
    # Copy file so artifacts in CI propogate without overwriting
    shutil.copyfile('gcov.tar.gz',os.path.join(petsc_arch,'gcov.tar.gz'))
    print("""Tarball created in %s"""%(petsc_dir))
    os.chdir(curdir)
    return

def make_htmlpage(gcov_dir,petsc_dir,LOC,tarballs,isCI):

    # Create index_gcov webpages using information processed from
    # running gcov
    # This is done in four stages
    # Stage 1: Extract tar balls,merge files and dump .lines files in gcov_dir
    # Stage 2: Process .lines files
    # Stage 3: Create marked HTML source code files
    # Stage 4: Create HTML pages having statistics and hyperlinks to HTML source code           files (files are sorted by filename and percentage code tested)
    #  Stores the main HTML pages in LOC if LOC is defined via command line argument o-wise it uses the default PETSC_DIR

    if os.path.isdir(gcov_dir):
        shutil.rmtree(gcov_dir)
    os.makedirs(gcov_dir)
    cwd = os.getcwd()
    # -------------------------- Stage 1 -------------------------------
    len_tarballs = len(tarballs)

    print("%s tarballs found\n%s" %(len_tarballs,tarballs))
    print("Extracting gcov directories from tar balls")
    #  Each tar file consists of a bunch of *.line files NOT inside a directory
    tmp_dirs = []
    for i in range(0,len_tarballs):
        tmp = []
        dir = os.path.join(gcov_dir,str(i))
        tmp.append(dir)
        os.mkdir(dir)
        os.system("cd "+dir+";gunzip -c "+tarballs[i] + "|tar -xof -")
        tmp.append(len(os.listdir(dir)))
        tmp_dirs.append(tmp)

    # each list in tmp_dirs contains the directory name and number of files in it
    # Cases to consider for gcov
    # 1) Gcov runs fine on all machines = Equal number of files in all the tarballs.
    # 2) Gcov runs fine on atleast one machine = Unequal number of files in the tarballs.The smaller tarballs are subset of the largest tarball(s)
    # 3) Gcov doesn't run correctly on any of the machines...possibly different files in tarballs

    # Case 2 is implemented for now...sort the tmp_dirs list in reverse order according to the number of files in each directory
    tmp_dirs.sort(key=operator.itemgetter(1),reverse=True)

    # Create temporary gcov directory to store .lines files
    print("Merging files")
    nfiles = tmp_dirs[0][1]
    files_dir1 = os.listdir(tmp_dirs[0][0])
    print(files_dir1)
    for i in range(0,nfiles):
        out_file = os.path.join(gcov_dir,files_dir1[i])
        out_fid  = open(out_file,'w')

        in_file = os.path.join(tmp_dirs[0][0],files_dir1[i])
        in_fid = open(in_file,'r')
        lines = in_fid.readlines()
        in_fid.close()
        for j in range(1,len(tmp_dirs)):
            in_file = os.path.join(tmp_dirs[j][0],files_dir1[i])
            try:
                in_fid = open(in_file,'r')
            except IOError:
                continue
            new_lines = in_fid.readlines()
            lines = list(set(lines)&set(new_lines)) # Find intersection
            in_fid.close()

        if(len(lines) != 0):
            lines.sort(key=int)
            out_fid.writelines(lines)
            out_fid.flush()

        out_fid.close()

    # Remove directories created by extracting tar files
    print("Removing temporary directories")
    for j in range(0,len(tmp_dirs)):
        shutil.rmtree(tmp_dirs[j][0])

    # ------------------------- End of Stage 1 ---------------------------------

    # ------------------------ Stage 2 -------------------------------------
    print("Processing .lines files in %s" %(gcov_dir))
    gcov_filenames = os.listdir(gcov_dir)
    nsrc_files = 0;
    nsrc_files_not_tested = 0;
    src_not_tested_path = [];
    src_not_tested_filename = [];
    src_not_tested_lines = [];
    src_not_tested_nlines = [];
    ctr = 0;
    print("Processing gcov files")
    for file in gcov_filenames:
        tmp_filename = file.replace('_',os.sep)
        src_file = tmp_filename.split('.lines')[0]
        gcov_file = gcov_dir+os.sep+file
        gcov_fid = open(gcov_file,'r')
        nlines_not_tested = 0
        lines_not_tested = []
        for line in gcov_fid:
            nlines_not_tested += 1
            temp_line1 = line.lstrip()
            temp_line2 = temp_line1.strip('\n')
            lines_not_tested.append(temp_line2)
        if nlines_not_tested :
            nsrc_files_not_tested += 1
            k = src_file.rfind(os.sep)
            src_not_tested_filename.append(src_file[k+1:])
            src_not_tested_path.append(src_file[:k])
            src_not_tested_lines.append(lines_not_tested)
            src_not_tested_nlines.append(nlines_not_tested)
        nsrc_files += 1
        gcov_fid.close()

    # ------------------------- End of Stage 2 --------------------------

    # ---------------------- Stage 3 -----------------------------------
    print("Creating marked HTML files")
    temp_string = '<a name'
    spaces_12 = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp'
    file_len = len(src_not_tested_nlines)
    fileopen_error = [];
    ntotal_lines = 0
    ntotal_lines_not_tested = 0
    output_list = []
    nfiles_not_processed = 0
    sep = LOC+os.sep
    for file_ctr in range(0,file_len):
        inhtml_file = petsc_dir+os.sep+src_not_tested_path[file_ctr]+os.sep+src_not_tested_filename[file_ctr]+'.html'
        outhtml_file = LOC+os.sep+src_not_tested_path[file_ctr]+os.sep+src_not_tested_filename[file_ctr]+'.gcov.html'
        try:
            inhtml_fid = open(inhtml_file,"r")
        except IOError:
            # Error check for files not opened correctly or file names not parsed correctly in stage 1
            fileopen_error.append([src_not_tested_path[file_ctr],src_not_tested_filename[file_ctr]])
            nfiles_not_processed += 1
            continue
        temp_list = []
        temp_list.append(src_not_tested_filename[file_ctr])
        temp_list.append(outhtml_file.split(sep)[1]) # Relative path of hyperlink
        temp_list.append(src_not_tested_nlines[file_ctr])

        outhtml_fid = open(outhtml_file,"w")
        lines_not_tested = src_not_tested_lines[file_ctr]
        nlines_not_tested = src_not_tested_nlines[file_ctr]
        line_ctr = 0
        last_line_blank = 0
        src_line = 0
        for line_temp in inhtml_fid:
            pre_issue_fix = 0
            line = line_temp.split('\n')[0]
            if(line.find(temp_string) != -1):
                nsrc_lines = int(line.split(':')[0].split('line')[1].split('"')[0].lstrip())
                src_line = 1;
            if (line_ctr < nlines_not_tested):
                temp_line = 'line'+src_not_tested_lines[file_ctr][line_ctr]
                if (line.find(temp_line) != -1):
                    # Untested line
                    if(line.startswith('<pre width=')):
                        num = line.find('>')
                        temp_outline = line[:num+1]+'<font color="red">Untested :&nbsp;&nbsp;</font>'+line[num+1:]
                    else:
                        temp_outline = '<font color="red">Untested :&nbsp;&nbsp;</font>'+line

                    line_ctr += 1
                else:
                    if(line.startswith('<pre width=')):
                        pre_issue_fix = 1;
                        num = line.find('>')
                    if(line.find(temp_string) != -1):
                        line_num = int(line.split(':')[0].split('line')[1].split('"')[0].lstrip())

                        if (line_num > int(src_not_tested_lines[file_ctr][line_ctr])):
                            while (int(src_not_tested_lines[file_ctr][line_ctr]) < line_num):
                                line_ctr += 1
                                if(line_ctr == nlines_not_tested):
                                    last_line_blank = 1
                                    temp_outline = spaces_12+line
                                    break
                            if (last_line_blank == 0):
                                temp_line = 'line'+src_not_tested_lines[file_ctr][line_ctr]
                                if(line.find(temp_line) != -1):
                                    temp_outline =  '<font color="red">Untested :&nbsp;&nbsp</font>'+line
                                    line_ctr += 1
                                else:
                                    if pre_issue_fix:
                                        temp_outline = line[:num+1]+spaces_12+line[num+1:]
                                    else:
                                        temp_outline = spaces_12+line
                        else:
                            if pre_issue_fix:
                                temp_outline = line[:num+1]+spaces_12+line[num+1:]
                            else:
                                temp_outline = spaces_12+line
                    else:
                        temp_outline = spaces_12+line
            else:
                temp_outline = spaces_12+line
            print(temp_outline, file=outhtml_fid)
            outhtml_fid.flush()

        inhtml_fid.close()
        outhtml_fid.close()

        ntotal_lines += nsrc_lines
        ntotal_lines_not_tested += src_not_tested_nlines[file_ctr]
        per_code_not_tested = float(src_not_tested_nlines[file_ctr])/float(nsrc_lines)*100.0

        temp_list.append(nsrc_lines)
        temp_list.append(per_code_not_tested)

        output_list.append(temp_list)

    shutil.rmtree(gcov_dir)
    # ------------------------------- End of Stage 3 ----------------------------------------

    # ------------------------------- Stage 4 ----------------------------------------------
    # Create Main HTML page containing statistics and marked HTML file links
    print("Creating main HTML page")
    # Create the main html file
    # ----------------------------- index_gcov1.html has results sorted by file name ----------------------------------
    # ----------------------------- index_gcov2.html has results sorted by % code tested ------------------------------
    date_time = strftime("%x %X %Z")
    outfile_name1 = LOC+os.sep+'index_gcov1.html'
    outfile_name2 = LOC+os.sep+'index_gcov2.html'
    out_fid = open(outfile_name1,'w')
    print("""<html>
    <head>
      <title>PETSc:Code Testing Statistics</title>
    </head>
    <body style="background-color: rgb(213, 234, 255);">""", file=out_fid)
    print("""<center>%s</center>"""%(date_time), file=out_fid)
    print("""<h2><center>Gcov statistics </center></h2>""", file=out_fid)
    print("""<center><font size = "4">Number of source code files = %s</font></center>""" %(nsrc_files), file=out_fid)
    print("""<center><font size = "4">Number of source code files not tested fully = %s</font></center>""" %(nsrc_files_not_tested), file=out_fid)
    if float(nsrc_files) > 0: ratio = float(nsrc_files_not_tested)/float(nsrc_files)*100.0
    else: ratio = 0.0
    print("""<center><font size = "4">Percentage of source code files not tested fully = %3.2f</font></center><br>""" %(ratio), file=out_fid)
    print("""<center><font size = "4">Total number of source code lines = %s</font></center>""" %(ntotal_lines), file=out_fid)
    print("""<center><font size = "4">Total number of source code lines not tested = %s</font></center>""" %(ntotal_lines_not_tested), file=out_fid)
    if float(ntotal_lines) > 0: ratio = float(ntotal_lines_not_tested)/float(ntotal_lines)*100.0
    else: ratio = 0.0
    print("""<center><font size = "4">Percentage of source code lines not tested = %3.2f</font></center>""" %ratio, file=out_fid)
    print("""<hr>
    <a href = %s>See statistics sorted by percent code tested</a>""" % ('index_gcov2.html'), file=out_fid)
    print("""<br><br>
    <h4><u><center>Statistics sorted by file name</center></u></h4>""", file=out_fid)
    print("""<table border="1" align = "center">
    <tr><th>Source Code</th><th>Lines in source code</th><th>Number of lines not tested</th><th>% Code not tested</th></tr>""", file=out_fid)

    output_list.sort(key=lambda x:x[0].lower())
    for file_ctr in range(0,nsrc_files_not_tested-nfiles_not_processed):
        print("<tr><td><a href = %s>%s</a></td><td>%s</td><td>%s</td><td>%3.2f</td></tr>" % (output_list[file_ctr][1],output_list[file_ctr][0],output_list[file_ctr][3],output_list[file_ctr][2],output_list[file_ctr][4]), file=out_fid)

    print("""</body>
    </html>""", file=out_fid)
    out_fid.close()

    # ----------------------------- index_gcov2.html has results sorted by percentage code tested ----------------------------------
    out_fid = open(outfile_name2,'w')
    print("""<html>
    <head>
      <title>PETSc:Code Testing Statistics</title>
    </head>
    <body style="background-color: rgb(213, 234, 255);">""", file=out_fid)
    print("""<center>%s</center>"""%(date_time), file=out_fid)
    print("""<h2><center>Gcov statistics</center></h2>""", file=out_fid)
    print("""<center><font size = "4">Number of source code files = %s</font></center>""" %(nsrc_files), file=out_fid)
    print("""<center><font size = "4">Number of source code files not tested fully = %s</font></center>""" %(nsrc_files_not_tested), file=out_fid)
    if float(nsrc_files) > 0: ratio = float(nsrc_files_not_tested)/float(nsrc_files)*100.0
    else: ratio = 0.0
    print("""<center><font size = "4">Percentage of source code files not tested fully = %3.2f</font></center><br>""" %ratio, file=out_fid)
    print("""<center><font size = "4">Total number of source code lines = %s</font></center>""" %(ntotal_lines), file=out_fid)
    print("""<center><font size = "4">Total number of source code lines not tested = %s</font></center>""" %(ntotal_lines_not_tested), file=out_fid)
    if float(ntotal_lines) > 0: ratio = float(ntotal_lines_not_tested)/float(ntotal_lines)*100.0
    else: ratio = 0.0
    print("""<center><font size = "4">Percentage of source code lines not tested = %3.2f</font></center>""" % ratio, file=out_fid)
    print("""<hr>
    <a href = %s>See statistics sorted by file name</a>""" % ('index_gcov1.html'), file=out_fid)
    print("""<br><br>
    <h4><u><center>Statistics sorted by percent code tested</center></u></h4>""", file=out_fid)
    print("""<table border="1" align = "center">
    <tr><th>Source Code</th><th>Lines in source code</th><th>Number of lines not tested</th><th>% Code not tested</th></tr>""", file=out_fid)
    output_list.sort(key=operator.itemgetter(4),reverse=True)
    for file_ctr in range(0,nsrc_files_not_tested-nfiles_not_processed):
        print("<tr><td><a href = %s>%s</a></td><td>%s</td><td>%s</td><td>%3.2f</td></tr>" % (output_list[file_ctr][1],output_list[file_ctr][0],output_list[file_ctr][3],output_list[file_ctr][2],output_list[file_ctr][4]), file=out_fid)

    print("""</body>
    </html>""", file=out_fid)
    out_fid.close()

    print("End of gcov script")
    print("""See index_gcov1.html in %s""" % (LOC))
    return

def main():

    parser = optparse.OptionParser(usage="%prog [options] ")
    parser.add_option('-p', '--petsc_dir', dest='petsc_dir',
                      help='Location of PETSC_DIR',
                      default='')
    parser.add_option('-a', '--petsc_arch', dest='petsc_arch',
                      help='Location of PETSC_ARCH',
                      default='')
    parser.add_option('-l', '--loc', dest='loc',
                      help='Location',
                      default='')
    parser.add_option('-r', '--run_gcov', dest='run_gcov',
                      help='Running gcov and printing tarball',
                      action='store_true',default=False)
    parser.add_option('-m', '--merge_gcov', dest='merge_gcov',
                      help='Merging gcov results and creating main html page',
                      action='store_true',default=False)
    options, args = parser.parse_args()

    if 'USER' in os.environ:
      USER = os.environ['USER']
    else:
      USER = 'petsc_ci'
    gcov_dir = "/tmp/gcov-"+USER

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
        print("Running gcov and creating tarball")
        run_gcov(gcov_dir,petsc_dir,petsc_arch)
        make_tarball(gcov_dir,petsc_dir,petsc_arch)
    elif options.merge_gcov:
        print("Creating main html page")
        # check to see if LOC is given
        if options.loc:
            print("Using %s to save the main HTML file pages" % (options.loc))
            LOC = options.loc
        else:
            print("No Directory specified for saving main HTML file pages, using PETSc root directory")
            LOC = petsc_dir

        tarballs = glob.glob(os.path.join(LOC,'*.tar.gz'))

        # Gitlab CI organizes things differently
        isCI=False
        if len(tarballs)==0:
          tarballs=glob.glob(os.path.join(LOC,'arch-*/gcov.tar.gz'))
          isCI=True

        if len(tarballs)==0:
          print("No coverage tarballs found")
          return

        make_htmlpage(gcov_dir,petsc_dir,LOC,tarballs,isCI)
    else:
        parser.print_usage()

if __name__ == '__main__':
    main()

