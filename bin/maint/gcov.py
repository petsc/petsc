#!/usr/bin/python

# Source code for creating marked HTML files from information processed from
# running gcov
# This is done in three stages
# Stage 1: Process lines files storing gcov information
# Stage 2: Create marked HTML source code files
# Stage 3: Create HTML page having statistics and hyperlinks to HTML source code           files (files are sorted by filename and percentage code tested) 

import os
import string
import operator

# ----------------- Stage 1 ---------------
PETSC_DIR = os.environ['PETSC_DIR']
gcov_dir = PETSC_DIR+'/tmp/gcov'
gcov_filenames = os.listdir(gcov_dir)
nsrc_files = 0; 
nsrc_files_not_tested = 0;
src_not_tested_path = [];
src_not_tested_filename = [];
src_not_tested_lines = [];
src_not_tested_nlines = [];
ctr = 0;
print "Processing gcov files ...."
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

print "Finished processing gcov files"
# ------------------------- End of Stage 1 --------------------------

# ---------------------- Stage 2 -----------------------------------
print "Creating marked HTML files ...."
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
        # One file bit_mask.c has an underscore in its name and hence parsing in stage 1 gives an error
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
#    print >>out_fid,"<tr><td><a href = %s>%s</a></td><td>%s</td><td>%s</td><td>%3.2f</td></tr>" % (outhtml_file,src_not_tested_filename[file_ctr],nsrc_lines,src_not_tested_nlines[file_ctr],per_code_not_tested)
#    out_fid.flush()
print "Finished creating marked HTML files"
# ------------------------------- End of Stage 2 ----------------------------------------

# ------------------------------- Stage 3 ----------------------------------------------
# Main HTML page containts statistics and individual file results
print "Creating main HTML page ...."
# Create the main html file                                                                                                                                    
# ----------------------------- index_gcov1.html has results sorted by file name ----------------------------------
# ----------------------------- index_gcov2.html has results sorted by % code tested ------------------------------
outfile_name1 = PETSC_DIR+'/'+'index_gcov1.html'
outfile_name2 = PETSC_DIR+'/'+'index_gcov2.html'
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

print "Finished creating main HTML page"
print """See index_gcov1.html in %s""" % (PETSC_DIR)

