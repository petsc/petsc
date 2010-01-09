#!/usr/bin/python

# This script combines $PETSC_DIR/makefile script 'allgcov' and conf/rules script 'gcov'

import os
import string
import shutil

PETSC_DIR = os.environ['PETSC_DIR']
USER = os.environ['USER']
# Create temp directory to store .lines files
gcov_dir = "/tmp/gcov-"+USER
if os.path.isdir(gcov_dir):
    shutil.rmtree(gcov_dir)
os.mkdir(gcov_dir)
for root,dirs,files in os.walk(os.path.join(PETSC_DIR,"src")):
# exclude tests and tutorial directories
    if root.endswith('tests') | root.endswith('tutorials'):
        continue
    os.chdir(root)
    for file_name in files:
        csrc_file = file_name.endswith('.c')
        if csrc_file:
            c_file = file_name.split('.c')[0]
            gcov_graph_file = c_file+".gcno"
            gcov_data_file  = c_file+".gcda" 
            if os.path.isfile(os.path.join(gcov_graph_file)) & os.path.isfile(os.path.join(gcov_data_file)):
                # gcov created .gcno and .gcda files => create .gcov file,parse it and save the untested code line
                # numbers in .lines file
                os.system("gcov "+file_name)
                gcov_file = file_name+".gcov"
                try:
                    gcov_fid = open(gcov_file,'r')
                    root_tmp1 = 'src'+root.split("src")[1].replace(os.sep,'_')
                    lines_fid = open(os.path.join(gcov_dir,root_tmp1+'_'+file_name+'.lines'),'w')
                    for line in gcov_fid:
                        if line.find("#####") > 0:
                            line_num = line.split(":")[1].strip()
                            print >>lines_fid,"""%s"""%(line_num)
                    gcov_fid.close()
                    lines_fid.close()
                except IOError:
                    continue
            else:
                # gcov did not create .gcno or .gcda file,save the source code line numbers to .lines file
                file_id = open(file_name,'r')
                root_tmp1 = 'src'+root.split("src")[1].replace(os.sep,'_')
                lines_fid = open(os.path.join(gcov_dir,root_tmp1+'_'+file_name+'.lines'),'w')
                nlines = 0
                line_num = 1
                for line in file_id:
                    if line.strip() == '':
                        line_num += 1
                    else:
                        print >>lines_fid,"""%s"""%(line_num)
                        line_num += 1
                file_id.close()
                lines_fid.close()

os.chdir(gcov_dir)
# Create tarball
os.system("tar -czf "+PETSC_DIR+"/gcov.tar.gz *.lines")
shutil.rmtree(gcov_dir)
