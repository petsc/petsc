#!/usr/bin/python

# Source code for creating marked HTML files from information processed from
# running gcov
# This is done in four stages
# Stage 1: Extract tar balls,merge files and dump .lines files in $PETSC_DIR/tmp/gcov
# Stage 2: Process .lines files
# Stage 3: Create marked HTML source code files
# Stage 4: Create HTML pages having statistics and hyperlinks to HTML source code           files (files are sorted by filename and percentage code tested) 
#  Stores the main HTML pages in LOC if LOC is defined via command line argument o-wise it uses the default PETSC_DIR
import os
import string
import operator
import sys
import shutil
import glob

PETSC_DIR = os.environ['PETSC_DIR']
gcov_dir = PETSC_DIR+'/tmp/gcov'
cwd = os.getcwd()
# -------------------------- Stage 1 -------------------------------
tarballs = glob.glob('*.gz')
len_tarballs = len(tarballs)
if len_tarballs == 0:
    print "No gcov tar balls found in directory %s" %(cwd)
    sys.exit()

print "%s tarballs found\n%s" %(len_tarballs,tarballs)
print "Extracting gcov directories from tar balls"
tmp_dirs = []
for i in range(0,len_tarballs):
    tmp = []
    dir_name = tarballs[i].split('.tar')[0]
    tmp.append(dir_name)
    os.system("gunzip -c" +" "+  tarballs[i] + "|tar -xof -")
    dir = cwd+'/'+dir_name
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
os.system("mkdir"+" "+gcov_dir)
print "Merging files"
nfiles = tmp_dirs[0][1]
dir1 = cwd+'/'+tmp_dirs[0][0]
files_dir1 = os.listdir(dir1)
for i in range(0,nfiles):
    out_file = gcov_dir+'/'+files_dir1[i]
    out_fid  = open(out_file,'w')

    in_file = tmp_dirs[0][0]+'/'+files_dir1[i]
    in_fid = open(in_file,'r')
    lines = in_fid.readlines()
    in_fid.close()
    for j in range(1,len(tmp_dirs)):
        in_file = tmp_dirs[j][0]+'/'+files_dir1[i]
        try:
            in_fid = open(in_file,'r')
        except IOError:
            print "Did not find file %s in directory %s" %(files_dir1[i],tmp_dirs[j][0])
            continue
        new_lines = in_fid.readlines()
        lines = list(set(lines)&set(new_lines)) # Find intersection             
        in_fid.close()

    if(len(lines) != 0):
	lines.sort()
        out_fid.writelines(lines)
        out_fid.flush()

    out_fid.close()

# Remove directories created by extracting tar files                                                                                 
print "Removing temporary directories"
for j in range(0,len(tmp_dirs)):
    shutil.rmtree(tmp_dirs[j][0])

# ------------------------- End of Stage 1 ---------------------------------

# ------------------------ Stage 2 -------------------------------------
print "Processing .lines files in %s" %(gcov_dir)
gcov_filenames = os.listdir(gcov_dir)
nsrc_files = 0; 
nsrc_files_not_tested = 0;
src_not_tested_path = [];
src_not_tested_filename = [];
src_not_tested_lines = [];
src_not_tested_nlines = [];
ctr = 0;
print "Processing gcov files"
for file in gcov_filenames:
    gcov_file = file
    gcov_file = PETSC_DIR+'/'+string.replace(gcov_file,'_','/')
    src_file = string.strip(gcov_file,'.lines')
    gcov_file = gcov_dir+'/'+file
    gcov_fid = open(gcov_file,'r')
    nlines_not_tested = 0
    lines_not_tested = []
    for line in gcov_fid:
        nlines_not_tested += 1
        temp_line1 = line.lstrip()  # Strings are immutable!! cannot perform operation on a string and save it as the same string 
        temp_line2 = temp_line1.strip('\n')
        lines_not_tested.append(temp_line2)
    if nlines_not_tested :   
        nsrc_files_not_tested += 1
        k = string.rfind(src_file,'/')
        src_not_tested_filename.append(src_file[k+1:])
        src_not_tested_path.append(src_file[:k])
        src_not_tested_lines.append(lines_not_tested)
        src_not_tested_nlines.append(nlines_not_tested)
    nsrc_files += 1
    gcov_fid.close()

# ------------------------- End of Stage 2 --------------------------

# ---------------------- Stage 3 -----------------------------------
print "Creating marked HTML files"
temp_string = '<a name'
file_len = len(src_not_tested_nlines)
fileopen_error = [];
ntotal_lines = 0
ntotal_lines_not_tested = 0
output_list = []
nfiles_not_processed = 0

for file_ctr in range(0,file_len):
    inhtml_file = src_not_tested_path[file_ctr]+'/'+src_not_tested_filename[file_ctr]+'.html'
    outhtml_file = src_not_tested_path[file_ctr]+'/'+src_not_tested_filename[file_ctr]+'.gcov.html'
    try:
        inhtml_fid = open(inhtml_file,"r")
    except IOError:
        # Error check for files not opened correctly or file names not parsed correctly in stage 1
        fileopen_error.append([src_not_tested_path[file_ctr],src_not_tested_filename[file_ctr]])
        nfiles_not_processed += 1
        continue

    temp_list = []
    temp_list.append(src_not_tested_filename[file_ctr])
    temp_list.append(outhtml_file)
    temp_list.append(src_not_tested_nlines[file_ctr])

    outhtml_fid = open(outhtml_file,"w")
    lines_not_tested = src_not_tested_lines[file_ctr]
    nlines_not_tested = src_not_tested_nlines[file_ctr]
    line_ctr = 0
    last_line_blank = 0
    for line in inhtml_fid:
        if(line.find(temp_string) != -1):
            nsrc_lines = int(line.split(':')[0].split('line')[1].split('"')[0].lstrip())
        if (line_ctr < nlines_not_tested):
            temp_line = 'line'+src_not_tested_lines[file_ctr][line_ctr]
            if (line.find(temp_line) != -1):
                temp_outline = '<table><tr><td bgcolor="yellow">'+'<font size="4" color="red">!</font>'+line+'</td></table>'
                line_ctr += 1
            else:
                # Gcov information contains blank line numbers which C2HTML doesn't print, Need to handle this
                # Marked line numbers 
                if(line.find(temp_string) != -1):
                    line_num = int(line.split(':')[0].split('line')[1].split('"')[0].lstrip())

                    if (line_num > int(src_not_tested_lines[file_ctr][line_ctr])):
                        while (int(src_not_tested_lines[file_ctr][line_ctr]) < line_num):
                            line_ctr += 1
                            if(line_ctr == nlines_not_tested):
                                last_line_blank = 1
                                temp_outline = line
                                break
                        if (last_line_blank == 0):        
                            temp_line = 'line'+src_not_tested_lines[file_ctr][line_ctr]
                            if(line.find(temp_line) != -1):
                                temp_outline =  '<table><tr><td bgcolor="yellow">'+'<font size="4" color="red">!</font>'+line+'</td></table>'
                                line_ctr += 1
                            else:
                                temp_outline = line
                    else:
                        temp_outline = line
                else:    
                    temp_outline = line
        else:
            temp_outline = line

        print >>outhtml_fid,temp_outline
        outhtml_fid.flush()

    inhtml_fid.close()
    outhtml_fid.close()

    ntotal_lines += nsrc_lines
    ntotal_lines_not_tested += src_not_tested_nlines[file_ctr]
    per_code_not_tested = float(src_not_tested_nlines[file_ctr])/float(nsrc_lines)*100.0
     
    temp_list.append(nsrc_lines)
    temp_list.append(per_code_not_tested)

    output_list.append(temp_list)

# Don't need $PETSC_DIR/tmp/gcov,remove it
shutil.rmtree(gcov_dir)
# ------------------------------- End of Stage 3 ----------------------------------------

# ------------------------------- Stage 4 ----------------------------------------------
# Create Main HTML page containing statistics and marked HTML file links
print "Creating main HTML page"
# Create the main html file                                                                                                                                    
# ----------------------------- index_gcov1.html has results sorted by file name ----------------------------------
# ----------------------------- index_gcov2.html has results sorted by % code tested ------------------------------
# check to see if LOC is given
if (len(sys.argv) == 2):
    print "Using %s to save the main HTML file pages" % (sys.argv[1])
    LOC = sys.argv[1]
else:
    print "No Directory specified for saving main HTML file pages, using PETSc root directory"
    LOC = PETSC_DIR

outfile_name1 = LOC+'/'+'index_gcov1.html'
outfile_name2 = LOC+'/'+'index_gcov2.html'
out_fid = open(outfile_name1,'w')                                            
print >>out_fid, \
"""<html>                                                                                                                                                      
<head>                                                                                                                                                         
  <title>PETSc:Code Testing Statistics</title>                                                                                                               
</head>                                                                                                                                                        
<body style="background-color: rgb(213, 234, 255);">                                                                                                           
<h2><center>Gcov statistics </center></h2>"""
print >>out_fid,"""<center><font size = "4">Number of source code files = %s</font></center>""" %(nsrc_files)
print >>out_fid,"""<center><font size = "4">Number of source code files not tested fully= %s</font></center>""" %(nsrc_files_not_tested)
print >>out_fid,"""<center><font size = "4">Percentage of source code files not tested fully = %3.2f</font></center><br>""" % (float(nsrc_files_not_tested)/float(nsrc_files)*\
100.0)
print >>out_fid,"""<center><font size = "4">Total number of source code lines = %s</font></center>""" %(ntotal_lines)
print >>out_fid,"""<center><font size = "4">Total number of source code lines not tested = %s</font></center>""" %(ntotal_lines_not_tested)
print >>out_fid,"""<center><font size = "4">Percentage of source code lines not tested = %3.2f</font></center>""" % (float(ntotal_lines_not_tested)/float(ntotal_lines)*\
100.0)
print >>out_fid,"""<hr>    
<a href = %s>See statistics sorted by percent code tested</a>""" % (outfile_name2)
print >>out_fid,"""<br><br>
<h4><u><center>Statistics sorted by file name</center></u></h4>"""                                                        
print >>out_fid,"""<table border="1" align = "center">                                                                                                                            
<tr><th>Source Code</th><th>Lines in source code</th><th>Number of lines not tested</th><th>% Code not tested</th></tr>"""

output_list.sort(key=operator.itemgetter(0),reverse=False)
for file_ctr in range(0,nsrc_files_not_tested-nfiles_not_processed):
    print >>out_fid,"<tr><td><a href = %s>%s</a></td><td>%s</td><td>%s</td><td>%3.2f</td></tr>" % (output_list[file_ctr][1],output_list[file_ctr][0],output_list[file_ctr][3],output_list[file_ctr][2],output_list[file_ctr][4])

print >>out_fid,"""</body>
</html>"""
out_fid.close()

# ----------------------------- index_gcov2.html has results sorted by percentage code tested ----------------------------------                                        
out_fid = open(outfile_name2,'w')                                                                                                    
print >>out_fid, \
"""<html>                                                                                                     
<head>                                                                                                                                                
  <title>PETSc:Code Testing Statistics</title>                                                             
</head> 
<body style="background-color: rgb(213, 234, 255);"> 
<h2><center>Gcov statistics</center></h2>"""
print >>out_fid,"""<center><font size = "4">Number of source code files = %s</font></center>""" %(nsrc_files)
print >>out_fid,"""<center><font size = "4">Number of source code files not tested fully= %s</font></center>""" %(nsrc_files_not_tested)
print >>out_fid,"""<center><font size = "4">Percentage of source code files not tested fully = %3.2f</font></center><br>""" % (float(nsrc_files_not_tested)/float(nsrc_files)*100.0)
print >>out_fid,"""<center><font size = "4">Total number of source code lines = %s</font></center>""" %(ntotal_lines)
print >>out_fid,"""<center><font size = "4">Total number of source code lines not tested = %s</font></center>""" %(ntotal_lines_not_tested)
print >>out_fid,"""<center><font size = "4">Percentage of source code lines not tested = %3.2f</font></center>""" % (float(ntotal_lines_not_tested)/float(ntotal_lines)*100.0)
print >>out_fid,"""<hr>
<a href = %s>See statistics sorted by file name</a>""" % (outfile_name1) 
print >>out_fid,"""<br><br>
<h4><u><center>Statistics sorted by percent code tested</center></u></h4>"""
print >>out_fid,"""<table border="1" align = "center">                                             
<tr><th>Source Code</th><th>Lines in source code</th><th>Number of lines not tested</th><th>% Code not tested</th></tr>"""
output_list.sort(key=operator.itemgetter(4),reverse=True)
for file_ctr in range(0,nsrc_files_not_tested-nfiles_not_processed):
    print >>out_fid,"<tr><td><a href = %s>%s</a></td><td>%s</td><td>%s</td><td>%3.2f</td></tr>" % (output_list[file_ctr][1],output_list[file_ctr][0],output_list[file_ctr][3],output_list[file_ctr][2],output_list[file_ctr][4])

print >>out_fid,"""</body>
</html>"""
out_fid.close()

print "End of gcov script"
print """See index_gcov1.html in %s""" % (LOC)

