#!/usr/bin/python
#
# Usage:
#       Run gcov on the results of "make alltests" and create tar ball containing coverage results for one machine
#           ./gcov.py -run_gcov
#       Generate html pages showing coverage by merging tar balls from multiple machines (index_gcov1.html and index_gcov2.html)
#           ./gcov.py -merge_gcov [LOC] tarballs
#

from __future__ import print_function
import os
import string
import shutil
import operator
import sys
from   time import gmtime,strftime

PETSC_DIR = os.environ['PETSC_DIR']

def run_gcov(gcov_dir):

    # 1. Runs gcov
    # 2. Saves the untested source code line numbers in
    #    xxx.c.lines files in gcov_dir

    print("Creating directory to save .lines files\n")
    if os.path.isdir(gcov_dir):
        shutil.rmtree(gcov_dir)
    os.mkdir(gcov_dir)
    print("Running gcov\n")
    for root,dirs,files in os.walk(os.path.join(PETSC_DIR,"src")):
        # Directories to skip
        if (root.find('tests') != -1) | (root.find('tutorials') != -1) | (root.find('benchmarks') != -1)| (root.find('examples') != -1) | (root.find('src'+os.sep+'dm'+os.sep+'mesh') != -1) | (root.find('draw'+os.sep+'impls'+os.sep+'win32') != -1) | (root.find('impls'+os.sep+'python') != -1) :
            continue
        os.chdir(root)
        for file_name in files:
            csrc_file = file_name.endswith('.c')
            if csrc_file:
                c_file = file_name.split('.c')[0]
                PETSC_ARCH = os.environ['PETSC_ARCH']
                OBJDIR = os.path.join(PETSC_DIR, PETSC_ARCH, 'obj')
                objpath = os.path.join(OBJDIR, os.path.relpath(c_file, os.path.join(PETSC_DIR,"src")))
                gcov_graph_file = objpath+".gcno"
                gcov_data_file  = objpath+".gcda"
                if os.path.isfile(gcov_graph_file) and os.path.isfile(gcov_data_file):
                    # gcov created .gcno and .gcda files => create .gcov file,parse it and save the untested code line
                    # numbers in .lines file
                    os.system('gcov --object-directory "%s" "%s"' % (os.path.dirname(gcov_data_file), file_name))
                    gcov_file = file_name+".gcov"
                    try:
                        gcov_fid = open(gcov_file,'r')
                        root_tmp1 = root.split(PETSC_DIR+os.sep)[1].replace(os.sep,'_')
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
                    root_tmp1 = root.split(PETSC_DIR+os.sep)[1].replace(os.sep,'_')
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

def make_tarball(dirname):

    # Create tarball of .lines files stored in gcov_dir
    print("""Creating tarball in %s to store gcov results files""" %(PETSC_DIR))
    os.chdir(dirname)
    os.system("tar -czf "+PETSC_DIR+os.sep+"gcov.tar.gz *.lines")
    shutil.rmtree(dirname)
    print("""Tarball created in %s"""%(PETSC_DIR))
    return

def make_htmlpage(gcov_dir,LOC,tarballs):

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
    if len_tarballs == 0:
        print("No gcov tar balls found in directory %s" %(cwd))
        sys.exit()

    print("%s tarballs found\n%s" %(len_tarballs,tarballs))
    print("Extracting gcov directories from tar balls")
    #  Each tar file consists of a bunch of *.line files NOT inside a directory
    tmp_dirs = []
    for i in range(0,len_tarballs):
        tmp = []
        dir = os.path.join(gcov_dir,str(i))
        tmp.append(dir)
        os.mkdir(dir)
        os.system("cd "+dir+";gunzip -c "+cwd+os.sep+tarballs[i] + "|tar -xof -")
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
        tmp_filename = string.replace(file,'_',os.sep)
        src_file = string.split(tmp_filename,'.lines')[0]
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
            k = string.rfind(src_file,os.sep)
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
        inhtml_file = PETSC_DIR+os.sep+src_not_tested_path[file_ctr]+os.sep+src_not_tested_filename[file_ctr]+'.html'
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
        temp_list.append(string.split(outhtml_file,sep)[1]) # Relative path of hyperlink
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

    USER = os.environ['USER']
    gcov_dir = "/tmp/gcov-"+USER

    if (sys.argv[1] == "-run_gcov"):
        print("Running gcov and creating tarball")
        run_gcov(gcov_dir)
        make_tarball(gcov_dir)
    elif (sys.argv[1] == "-merge_gcov"):
        print("Creating main html page")
    # check to see if LOC is given
        if os.path.isdir(sys.argv[2]):
            print("Using %s to save the main HTML file pages" % (sys.argv[2]))
            LOC = sys.argv[2]
            tarballs = sys.argv[3:]
        else:
            print("No Directory specified for saving main HTML file pages, using PETSc root directory")
            LOC = PETSC_DIR
            tarballs = sys.argv[2:]

        make_htmlpage(gcov_dir,LOC,tarballs)
    else:
        print("No or invalid option specified:")
        print("Usage: To run gcov and create tarball")
        print("         ./gcov.py -run_gcov      ")
        print("Usage: To create main html page")
        print("         ./gcov.py -merge_gcov [LOC] tarballs")

if __name__ == '__main__':
    main()

