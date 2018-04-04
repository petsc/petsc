#!/usr/bin/python
#
#    See startnightly for the context of how this script is used
#
from __future__ import print_function
import os
import sys
import time
import re
import fnmatch
import dateutil.parser

## Early checks:

if len(sys.argv) < 4:
  print("Usage: $> runhtml.py BRANCH LOGDIR OUTFILE");
  print(" BRANCH  ... Branch log files to be processed");
  print(" LOGDIR  ... Directory where to find the log files");
  print(" OUTFILE ... The output file where the HTML code will be written to");
  print("Aborting...")
  sys.exit(1)


######### Packages to list ########

packages=["Chaco","CMake","CUDA","CUSP","Elemental","Exodusii","HDF5","Hypre","Metis","ML","MOAB","MUMPS","NetCDF","Pardiso","Parmetis","ptscotch","SPAI","STRUMPACK","Suitesparse","SuperLU","SuperLU_dist","cTetgen","triangle","ViennaCL"];



######### Helper routines #########
nowtime = dateutil.parser.parse(time.strftime('%a, %d %b %Y %H:%M:%S %z'))

# Helper function: Obtain execution time from log file:
def execution_time(logfilename):
  foundStart = False
  try:
    for line in open(logfilename):
      if not foundStart and re.search(r'^Starting (configure|make|test|examples) run', line):
        foundStart = True
        starttime = dateutil.parser.parse(line.split(' at ')[1])
      if re.search(r'^Finishing (configure|make|test|examples) run', line):
        endtime = dateutil.parser.parse(line.split(' at ')[1])
    exectime = endtime - starttime
  except:
    return 0
  try:
    agetime  = nowtime - starttime
  except:
    agetime  = nowtime.replace(tzinfo=None) - starttime
  if agetime.total_seconds() > 12*60*60:
    return -int(agetime.total_seconds())
  else:
    return int(exectime.total_seconds())

# Helper function: Convert number of seconds to format hh:mm:ss
def format_time(time_in_seconds):
  #print "time_in_seconds: " + str(time_in_seconds)
  time_string = "";
  if time_in_seconds < 0:
    time_string = "<td class=\"red\">";
    time_in_seconds = -time_in_seconds
  elif time_in_seconds > 1800:
    time_string = "<td class=\"yellow\">";
  else:
    time_string = "<td class=\"green\">";

  time_string += str(time_in_seconds / 60) + ":" + str(time_in_seconds % 60).zfill(2) + "</td>"

  return time_string;



###### Main execution body ##########


outfile = open(sys.argv[3], "w")
examples_summary_file = open(sys.argv[2] + "/examples_full_"+sys.argv[1]+".log", "w")

# Static HTML header:
outfile.write("""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head><title>PETSc Test Summary</title>
<style type="text/css">
div.main {
  max-width: 1500px;
  background: white;
  margin-left: auto;
  margin-right: auto;
  padding: 20px;
  padding-top: 0;
  border: 5px solid #CCCCCC;
  border-radius: 10px;
  background: #FBFBFB;
}
table {
  /*border: 1px solid black;
  border-radius: 10px;*/
  padding: 3px;
  margin-top: 0;
}
td a:link, td a:visited, td a:focus, td a:active {
  font-weight: bold;
  text-decoration: underline;
  color: black;
}
td a:hover {
  font-weight: bold;
  text-decoration: underline;
  color: black;
}
th {
  padding: 5px;
  padding-top: 5px;
  padding-bottom: 5px;
  font-size: 1.1em;
  font-weight: bold;
  text-align: center;
}
.verticalTableHeader {
    text-align:center;
    white-space:nowrap;
    transform-origin:50% 50%;
    -webkit-transform: rotate(-90deg);
    -moz-transform: rotate(-90deg);
    -ms-transform: rotate(-90deg);
    -o-transform: rotate(-90deg);
    transform: rotate(-90deg);
}
.verticalTableHeader p {
    margin:0 -100% ;
    display:inline-block;
    width:7px;
    font-size:0.75em;
}
.verticalTableHeader p:before{
    content:'';
    width:0;
    padding-top:110%;/* takes width as reference, + 10% for faking some extra padding */
    display:inline-block;
    vertical-align:middle;
}

td.desc {
  max-width: 650px;
  padding: 2px;
  font-size: 0.9em;
}
td.green {
  text-align: center;
  vertical-align: middle;
  padding: 2px;
  background: #01DF01;
  min-width: 30px;
}
td.yellow {
  text-align: center;
  vertical-align: middle;
  padding: 2px;
  background: #F4FA58;
  min-width: 30px;
}
td.red {
  text-align: center;
  vertical-align: middle;
  padding: 2px;
  background: #FE2E2E;
  min-width: 30px;
}
td.have {
  text-align: center;
  vertical-align: middle;
  background: #01DF01;
  padding:0;
}
td.centered {
  text-align: center;
  vertical-align: middle;
  padding:0;
}
</style>
</head>
<body><div class="main"> """)


outfile.write("<center><span style=\"font-size:1.3em; font-weight: bold;\">PETSc Test Summary</span><br />Last update: " + time.strftime('%a, %d %b %Y %H:%M:%S %z') + "</center>\n")

outfile.write("<center><table border=\"0\">\n");

outfile.write("<tr><th></th><th colspan=\"" + str(4+len(packages)) + "\">&nbsp;</td><th colspan=\"2\">Configure</th><th></th><th></th> <th colspan=\"3\">Make</th><th></th><th></th> <th colspan=\"2\">Examples</th></tr>\n");
outfile.write('<tr style="height:70px"></tr>');
outfile.write("<tr><th>Arch</th>");
outfile.write('<th class="verticalTableHeader"><p>Debug</p></th>');
outfile.write('<th class="verticalTableHeader"><p>Precision</p></th>');
outfile.write('<th class="verticalTableHeader"><p>Complex</p></th>');
outfile.write('<th class="verticalTableHeader"><p>Indices</p></th>');
outfile.write('<th class="verticalTableHeader"><p>MPICH-ErrorCheck</p></th>');
for package in packages:
 outfile.write('<th class="verticalTableHeader"><p>' + package + '</p></th>');
outfile.write("<th>Stat</th><th>Time</th><th></th><th></th> <th>Warn</th><th>Err</th><th>Time</th><th></th><th></th> <th>Prob?</th><th>Time</th><td><a href=\"examples_full_"+sys.argv[1]+".log\">[all]</a></td></tr>\n");

num_builds = 0

for root, dirs, filenames in os.walk(sys.argv[2]):
  filenames.sort()
  for f in filenames:
    if fnmatch.fnmatch(f, "build_" + sys.argv[1] + "_*.log"):

      num_builds += 1

      # form other file names:
      match = re.search("build_" + sys.argv[1] + "_arch-(.*).log", f)
      logfile_build = f
      logfile_build_full = os.path.join(root, f)
      logfile_make  = "make_"  + sys.argv[1] + "_arch-" + match.group(1) + ".log"
      logfile_make_full = os.path.join(root, logfile_make)
      logfile_examples  = "examples_"  + sys.argv[1] + "_arch-" + match.group(1) + ".log"
      logfile_examples_full = os.path.join(root, logfile_examples)
      logfile_configure = "configure_" + sys.argv[1] + "_arch-" + match.group(1) + ".log"
      logfile_configure_full = os.path.join(root, logfile_configure)

      print("Processing " + match.group(1))

      ### Start table row
      if match.group(1).find('linux-analyzer') >= 0:
        outfile.write("<tr><td>" + match.group(1) +
                      " <a href=\"analyzer-src_" + sys.argv[1] + ".log/index.html\"><FONT style=\"BACKGROUND-COLOR: orange\">[S]</FONT></a>" +
                      " <a href=\"analyzer-ex_" + sys.argv[1] + ".log/index.html\"><FONT style=\"BACKGROUND-COLOR: orange\">[E]</FONT></a>" +
                      "</td>")
      else:
        outfile.write("<tr><td>" + match.group(1) + "</td>")

      #
      ### Configure section
      #
      if not os.path.isfile(logfile_configure_full):
        outfile.write("<td class=\"red\" colspan=\"2\"><a href=\"" + logfile_build + "\">[build.log]</a></td>")
        continue

      # Checking for successful completion
      configure_success = False
      for line in open(logfile_configure_full):
        if line.startswith("Configure Options: "):
           configline = line.lower();
           # Debug:
           if configline.find("with-debugging=0") > 0:
             outfile.write("<td class=\"have\">N</td>");
           else:
             outfile.write("<td class=\"centered\">Y</td>");

           # Precision:
           if configline.find("--with-precision=single") > 0:
             outfile.write("<td class=\"have\">S</td>");
           elif configline.find("--with-precision=__float128") > 0:
             outfile.write("<td class=\"have\">Q</td>");
           else:
             outfile.write("<td class=\"centered\">D</td>");

           # Complex:
           if configline.find("with-scalar-type=complex") > 0:
             outfile.write("<td class=\"have\">Y</td>");
           else:
             outfile.write("<td class=\"centered\">N</td>");

           # Indices:
           if configline.find("with-64-bit-indices=1") > 0:
             outfile.write("<td class=\"have\">64</td>");
           else:
             outfile.write("<td class=\"centered\">32</td>");

           #MPICH-ErrorCheck
           if configline.find("enable-error-messages=all") > 0:
             outfile.write("<td class=\"have\">Y</td>");
           else:
             outfile.write("<td class=\"centered\">N</td>");

           # Packages:
           for package in packages:
             if configline.find(package.lower()) > 0:
               outfile.write("<td class=\"have\">Y</td>");
             else:
               outfile.write("<td class=\"centered\">N</td>");

        if re.search(r'Configure stage complete', line) or  re.search(r'Installation complete', line):
          outfile.write("<td class=\"green\">OK</td>")
          outfile.write(format_time(execution_time(logfile_configure_full)))
          configure_success = True

      if configure_success == False:
          outfile.write("<td class=\"red\">Fail</td>")
          outfile.write("<td class=\"red\">Fail</td>")
      outfile.write("<td><a href=\"" + logfile_configure + "\">[log]</a></td>")

      #
      # Check if some logs are missing. If so, don't process further and write 'incomplete' to table:
      #
      if not os.path.isfile(logfile_make_full) or not os.path.isfile(logfile_examples_full):
        print("  -- incomplete logs!")

        # Make/Build section
        outfile.write("<td></td>")
        outfile.write("<td class=\"red\" colspan=\"2\"><a href=\"" + logfile_build + "\">[build.log]</a></td>")
        if os.path.isfile(logfile_make_full):
          outfile.write("<td class=\"red\" colspan=\"2\"><a href=\"" + logfile_make + "\">[make.log]</a></td>")
        else:
          outfile.write("<td colspan=\"2\" class=\"red\">Incomplete</td>")

        # Examples section
        outfile.write("<td></td>")
        outfile.write("<td class=\"red\">Incomplete</td>")
        outfile.write("<td class=\"red\">Incomplete</td>")
        if os.path.isfile(logfile_examples_full): outfile.write("<td><a href=\"" + logfile_examples + "\">[log]</a></td>")
        else: outfile.write("<td></td>\n")
        outfile.write("</tr>\n");
        continue

      #
      ### Make section
      #

      outfile.write("<td></td>")
      # Warnings:
      warning_list = []
      exclude_warnings_re = ["unrecognized .pragma",
                          "warning: .SSL",
                          "warning: .BIO_",
                          "warning[s]* generated"]
      exclude_warnings = ["warning: statement not reached",
                          "warning: loop not entered at top",
                          "is deprecated",
                          "is superseded",
                          "Warning: You are using gcc version '4.8.4'. The version of gcc is not supported. The version currently supported with MEX is '4.7.x'.",
                          "warning: no debug symbols in executable (-arch x86_64)",
                          "(aka 'const double *') doesn't match specified 'MPI' type tag that requires 'double *'",
                          "(aka 'const int *') doesn't match specified 'MPI' type tag that requires 'int *'",
                          "(aka 'const long *') doesn't match specified 'MPI' type tag that requires 'long long *'",
                          "(aka 'long *') doesn't match specified 'MPI' type tag that requires 'long long *'",
                          "cusp/complex.h", "cusp/detail/device/generalized_spmv/coo_flat.h",
                          "Warning: Cannot tell what pointer points to, assuming global memory space",
                          "warning C4003: not enough actual parameters for macro 'PETSC_PASTE3_'",
                          "warning: linker scope was specified more than once",
                          "cl : Command line warning D9024 : unrecognized source file type",
                          "thrust/detail/vector_base.inl", "thrust/detail/tuple_transform.h", "detail/tuple.inl", "detail/launch_closure.inl"]
      for line in open(logfile_make_full):
        if re.search(r'[Ww]arning[: ]', line):
          has_serious_warning = True
          for warning in exclude_warnings_re:
            if re.search(warning, line):
              has_serious_warning = False
              break
          if has_serious_warning:
            for warning in exclude_warnings:
              if re.search(warning, line) or line.find(warning)>=0 :
                has_serious_warning = False
                break
          if has_serious_warning == True:
            warning_list.append(line)
      num_warnings = len(warning_list)
      if num_warnings > 0:
        outfile.write("<td class=\"yellow\">" + str(num_warnings) + "</td>")
      else:
        outfile.write("<td class=\"green\">" + str(num_warnings) + "</td>")

      # Errors:
      error_list = []
      error_list_with_context = []
      f = open(logfile_make_full)
      lines = f.readlines()
      for i in range(len(lines)):
        if re.search(" [Kk]illed", lines[i]) or re.search(" [Ff]atal[: ]", lines[i]) or re.search(" [Ee][Rr][Rr][Oo][Rr][: ]", lines[i]):
          error_list.append(lines[i])
          if i > 1:
            error_list_with_context.append(lines[i-2])
          if i > 0:
            error_list_with_context.append(lines[i-1])
          error_list_with_context.append(lines[i])
          if i+1 < len(lines):
            error_list_with_context.append(lines[i+1])
          if i+2 < len(lines):
            error_list_with_context.append(lines[i+2])
          if i+3 < len(lines):
            error_list_with_context.append(lines[i+3])
          if i+4 < len(lines):
            error_list_with_context.append(lines[i+4])
          if i+5 < len(lines):
            error_list_with_context.append(lines[i+5])

      num_errors = len(error_list)
      if num_errors > 0:
        outfile.write("<td class=\"red\">" + str(num_errors) + "</td>")
      else:
        outfile.write("<td class=\"green\">" + str(num_errors) + "</td>")
      outfile.write(format_time(execution_time(logfile_make_full)))
      outfile.write("<td><a href=\"filtered-" + logfile_make + "\">[log]</a><a href=\"" + logfile_make + "\">[full]</a></td>")

      # Write filtered output file:
      filtered_logfile = os.path.join(root, "filtered-" + logfile_make)
      filtered_file = open(filtered_logfile, "w")
      filtered_file.write("---- WARNINGS ----\n")
      for warning_line in warning_list:
        filtered_file.write(warning_line)
      filtered_file.write("\n---- ERRORS ----\n")
      for error_line in error_list_with_context:
        filtered_file.write(error_line)
      filtered_file.close()

      #
      ### Examples section
      #
      outfile.write("<td></td>")
      example_problem_num = 0
      for line in open(logfile_examples_full):
        examples_summary_file.write(line)
        if re.search(r'not ok', line):
          example_problem_num += 1
        if re.search(r'[Pp]ossible [Pp]roblem', line):
          example_problem_num += 1

      if example_problem_num < 1:
         outfile.write("<td class=\"green\">0</td>")
      else:
         outfile.write("<td class=\"yellow\">" + str(example_problem_num) + "</td>")
      outfile.write(format_time(execution_time(logfile_examples_full)))
      outfile.write("<td><a href=\"" + logfile_examples + "\">[log]</a></td>")

      ### End of row
      outfile.write("</tr>\n")

# write footer:
outfile.write("</table>")
outfile.write("Total number of builds: " + str(num_builds))
outfile.write("<br />This page is an automated summary of the output generated by the PETSc testsuite.<br /> It is generated by $PETSC_DIR/lib/petsc/bin/maint/runhtml.py.</center>\n")
outfile.write("</div></body></html>")
outfile.close()
examples_summary_file.close()

#print "Testing execution time: "
#print format_time(execution_time(sys.argv[2]))
