#!/usr/bin/python
##
## Generates a summary of all branches in the repository.
## Prints the branch creation date, the last update,
## and whether the respective branch has been merged to 'master' or 'next'
## Output is a standalone .html file.
##
## Usage: $> branches.py TYPE OUTPUT";
##   TYPE    ... Either 'all' or 'active'";
##   OUTFILE ... The output file where the HTML code will be written to";


from __future__ import print_function
import os
import sys
import time
import subprocess
import datetime

## Create a separate group in the summary for these:
branchnames = ["balay", "barry", "hzhang", "jed", "karlrupp", "knepley", "mark", "prbrune", "sarich", "stefano_zampini", "tisaac"]



## Worker function:

def printtable(outfile, branches, anchor):

  # Write table head:
  outfile.write("<center><a name=\"" + anchor + "\"></a>")
  outfile.write("<table>\n");
  outfile.write("<tr><th>Branchname</th><th>Created</th><th>Last Update</th><th>next  </th><th>master</th><th> </th></tr>\n");
  outfile.write("<tr><td>&nbsp;    </td><td>(days) </td><td>(days)     </td><td>&nbsp;</td><td>&nbsp;</td><td> </td></tr>\n");

  tablebuffer = [];

  # Write table body:
  for branch in branches:

    # Branch name
    tablerow = "<tr><td>" + branch[7:] +"</td>";

    # Created: First get commit hash of merge base, then get date for commit
    process = subprocess.Popen(["git", "merge-base", "remotes/" + branch, "master"], stdout=subprocess.PIPE)
    commithash, err = process.communicate()
    commithash = commithash.replace("\r", "").replace("\n", "")
    process = subprocess.Popen(["git", "log", "--pretty=format:%at", commithash, "-n", "1"], stdout=subprocess.PIPE)
    unixtime, err = process.communicate()
    num_days = (int(time.time()) - int(unixtime)) / (60*60*24)
    tablerow += "<td>" + str(int(num_days)) + "</td>";

    # Last Update:
    process = subprocess.Popen(["git", "log", "--pretty=format:%at", "remotes/" + branch, "-n", "1"], stdout=subprocess.PIPE)
    unixtime, err = process.communicate()
    num_days = (int(time.time()) - int(unixtime)) / (60*60*24)
    tablerow += "<td>" + str(int(num_days)) + "</td>";

    # Merged next:
    process = subprocess.Popen(["git", "log", "--pretty=oneline", "remotes/origin/next..remotes/" + branch, "-n", "1"], stdout=subprocess.PIPE)
    stdoutstring, err = process.communicate()
    if (len(stdoutstring) > 2):
      tablerow += "<td class=\"yellow\">no</td>"; 
    else:
      tablerow += "<td class=\"green\">yes</td>"; 

    # Merged master:
    process = subprocess.Popen(["git", "log", "--pretty=oneline", "remotes/origin/master..remotes/" + branch, "-n", "1"], stdout=subprocess.PIPE)
    stdoutstring, err = process.communicate()
    if (len(stdoutstring) > 2):
      tablerow += "<td class=\"red\">no</td>"; 
    else:
      tablerow += "<td class=\"green\">yes</td>"; 

    # Delete button
    #outfile.write("<td>Delete</td>")


    # End of row
    tablerow += "</tr>";
    tablebuffer.append([unixtime, tablerow]);

  # dump table:
  tablebuffer.sort(reverse=True);

  for row in tablebuffer:
    outfile.write(row[1]);
   
  outfile.write("</table></center>\n");

  


#####################################################
#### Start of main execution
#####################################################


## Early checks:

if len(sys.argv) < 3:
  print("Usage: $> branches.py TYPE OUTPUT");
  print(" TYPE    ... Either 'all' or 'active'");
  print(" OUTFILE ... The output file where the HTML code will be written to");
  print("Aborting...")
  sys.exit(1)

if (sys.argv[1] != "all" and sys.argv[1] != "active"):
  print("Unknown type: " + sys.argv[1]);
  print("Usage: $> branches.py TYPE OUTPUT");
  print(" TYPE    ... Either 'all' or 'active'");
  print(" OUTFILE ... The output file where the HTML code will be written to");
  print("Aborting...")
  sys.exit(1)


###### Main execution body ##########


outfile = open(sys.argv[2], "w")

# Static HTML header:
outfile.write("""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head><title>PETSc Branch Summary</title>
<style type="text/css">
div.main {
  max-width: 1300px;
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
  padding: 10px;
  padding-top: 5px;
  padding-bottom: 5px;
  font-size: 1.1em;
  font-weight: bold;
  text-align: center;
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
  min-width: 50px;
}
td.yellow {
  text-align: center;
  vertical-align: middle;
  padding: 2px;
  background: #F4FA58;
  min-width: 50px;
}
td.red {
  text-align: center;
  vertical-align: middle;
  padding: 2px;
  background: #FE2E2E;
  min-width: 50px;
}
</style>
</head>
<body><div class="main"> """)


outfile.write("<center><span style=\"font-size:1.3em; font-weight: bold;\">")
if (sys.argv[1] == "all"):
  outfile.write("Summary of ALL PETSc Branches")
else:
  outfile.write("Summary of ACTIVE PETSc Branches")
outfile.write("</span><br />Last update: " + time.strftime("%c") + "</center>\n")


# Build quicklink bar:
outfile.write("<br /><center>")
for branchprefix in branchnames:
  outfile.write("<a href=\"#" + branchprefix + "\">[" + branchprefix + "]</a>&nbsp;")
outfile.write("<a href=\"#other\">[other]</a>")
outfile.write("</center><br />")


# Get all remote branches
process = subprocess.Popen(["git", "branch", "-r"], stdout=subprocess.PIPE)
allbranches, err = process.communicate()

# Print branches for each of the branchnames defined above:
for branchprefix in branchnames:
  print("Working on " + branchprefix)
  branchprefix2 = "origin/" + branchprefix

  userbranches = []
  for line in allbranches.splitlines():
    if branchprefix2 in line:
      userbranches.append(line.strip())

  printtable(outfile, userbranches, branchprefix)

# Print all other branches by testing against the 'standard' branchnames:
print("Working on remaining branches")
otherbranches = []
for line in allbranches.splitlines():

  if "origin" not in line or "origin/HEAD" in line:
    continue

  is_other_branch = 1
  for branchprefix in branchnames:
    branchprefix2 = "origin/" + branchprefix
 
    if branchprefix2 in line:
      is_other_branch = 0

  if (is_other_branch == 1):
    otherbranches.append(line.strip())

printtable(outfile, otherbranches, "other")


# write footer:
outfile.write("</div></body></html>")
outfile.close()

  
