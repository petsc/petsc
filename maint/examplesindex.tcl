#!/usr/bin/env tclsh
# $Id: examplesindex.tcl,v 1.38 1999/02/03 00:19:05 balay Exp balay $ 

################################################
# This program scans the PETSc example files   #
# i.e. *.c, *.f, *.F and reads the formated    #
# comments in these examples, and writes the   #
# contents into files readable by MS-ACCESS    #
# It takes PETSC_DIR from env variables        #
# and the example dirs are src/*/examples      #
# and src/*/examples/tutorials                 #
#                                              #
# usage: examplesindex.tcl                     #
#                                              #
# options:                                     #
#    -noupdatewww  : also update the           #
#          wwwmanpages with links to examples  #
#                                              #
# purpose: To get cute tables which contain    #
# information like Concept:XXX is demonstrated #
# in ex1, ex2, ex3 etc..                       #
#                                              #
################################################





##################################################
####           processformat()               #####    
##################################################

proc processformat { filename buf } {
    global  Concepts Routines Processors Comment

    while { [string length [string trim $buf] ] != 0 } {
        # trim the buffer i.e Take away additional blanks or newline chars 
        set buf [ string trimleft  $buf ]

        # scan for keyword in the buffer which is delimited by ":"
        set start_data [ string first ":"  $buf ]
        set end_data   [ string first "\n" $buf ]
        
        if { $start_data == -1 } {
            puts stderr "$filename: Unable to scan the following text:"
            puts stderr $buf
            set Concepts($filename)   {}
            set Routines($filename)   {}
            set Processors($filename) {}
            set Comment($filename)    "" 
            return 0
        }

        set keyword   [ string range $buf 0  [ expr $start_data - 1] ]
        set dataline  [ string range $buf [ expr $start_data + 1 ]  [ expr $end_data - 1 ] ]
        set dataline  [ split $dataline ";" ]
        set buf       [ string range $buf [ expr $start_data + 1 ] end ]

        switch $keyword {
            Concepts {  
                set Concepts($filename) [ concat $Concepts($filename) $dataline ]
            }
            Routines {
                set Routines($filename) [ concat $Routines($filename) $dataline ]
            }
            Processors {
                set Processors($filename)  $dataline
            }
            Comment {
                # Comment can be spread out to multiple lines if so, it has to be
                # terminated by a "!" or it should be the the end of the file
                # use some temporary space here 
                
                set start_data [ string first ":"  $buf ]
                if { $start_data == -1 } {
                    #No more formated fileds, so all the text is comment
                    #set dataline  [ string trim $buf "\n" ]
                    set Comment($filename) $buf
                } else {
                    puts stderr "Skipping file: $filename"
                    puts stderr "Commnet field should be at the end"
                    set Concepts($filename)   {}
                    set Routines($filename)   {}
                    set Processors($filename) {}
                    set Comment($filename)    "" 
                }
                return 0
            }
            default {
                # Unknown Keyword, so abort
                puts stderr "Skipping file: $filename"
                puts stderr "Unknown Format: $keyword"
                set Concepts($filename)   {}
                set Routines($filename)   {}
                set Processors($filename) {}
                set Comment($filename)    "" 
                return 0
            }
        }
        set end_data   [ string first "\n" $buf ]
        set buf       [ string range $buf [ expr $end_data + 1 ] end ]
    }
    return 0
}

##################################################
####              scanfile()                 #####    
##################################################

proc scanfile { filename } {

    # open the file
    set fileid [ open $filename r ]
    set filebuff [ read $fileid ]

    # Find the start of the formated string i.e. "/*T"
    set start_comment [ string first /*T  $filebuff ]
    if { $start_comment == -1 } { 
        close $fileid
        return 1
    }
    
    # find the ending of the formated string i.e. "T*/"
    set end_comment [ string first T*/  $filebuff ]
    if { $end_comment == -1 } {
        set mesg "Incorrect format in file: $filename"
        puts stderr $mesg
        close fileid
        return 1
    }
    
    #extract the fromated string 
    set databuff [ string range $filebuff [ expr $start_comment + 3 ] [ expr $end_comment - 1 ]   ]

    # for fortran examples take away the additional comments 
    set suffix [ file extension $filename ]
    if { $suffix == ".f" || $suffix == ".F" } {
        regsub -all  "\nc|\nC|\n!" $databuff  "\n" databuff
    }

    # take care of the case where "; \n" might occur
        regsub -all "; *\n" $databuff  "\n" databuff


    # process the formated string
    processformat $filename $databuff
    close $fileid
    return 0
}


##################################################
####           deletespace()                 #####    
##################################################

proc deletespace { name } { 
    upvar $name buf 
    #Delete spaces from the list buf
    
    set n [ llength $buf ]
    set i 0
    set new {}
    while { $i < $n } {
        set temp [ join [ lindex $buf $i ] " " ]
        set temp  [string trim $temp ]
        lappend new $temp
        set i [ expr $i + 1 ]
        
    }
    set buf $new
    return 0
}

##################################################
####          delenelinefeed()               #####    
##################################################
proc deletelinefeed { name } { 
    upvar $name buf 
    #Delete spaces from the list buf
    
    set buf [join [ split $buf "\n" ] " " ]
}
##################################################
####       write_access_tables()             #####    
##################################################

proc write_access_tables { } {
    global  Concepts Routines Processors Comment PETSC_DIR_www files html

    set table1 [ open docs/tex/access/table1.txt  w ]
    puts $table1 "Key;Dir;Name;Source;Processors;comments"
    foreach filename $files {
        set tempdir [ file dirname $filename ]
        set tempfile [ file tail $filename ]
        set tempext [ file extension $filename ]
        set temphtml $html($filename)
        set mesg "$filename;$temphtml;$tempdir;$tempfile;$tempext;$Processors($filename);$Comment($filename)"
        puts  $table1 $mesg
    }
    close $table1
    exec /bin/chmod ug+w docs/tex/access/table1.txt
    
    
    set table2 [ open docs/tex/access/table2.txt w ]
    puts $table2 "Name of the File;Concepts"
    
    foreach filename $files {
        set n [ llength $Concepts($filename)  ]
        set temphtml $html($filename) 
        set i 0
        while { $i < $n } {
            set temp [ join [ lindex $Concepts($filename) $i ] " " ]
            set mesg "$temphtml;$temp"
            puts  $table2 $mesg
            set i [ expr $i + 1 ]
        }
    }
    close $table2
    exec /bin/chmod ug+w docs/tex/access/table2.txt
    
    set table3 [ open docs/tex/access/table3.txt w ]
>    puts $table3 "Name of the File;Routines"
    
    foreach filename $files {
        set n [ llength $Routines($filename)  ]
        set temphtml $html($filename) 
        set i 0
        while { $i < $n } {
            set temp [ join [ lindex $Routines($filename) $i ] " " ]
            set mesg "$temphtml;$temp"
            puts  $table3 $mesg
            set i [ expr $i + 1 ]
        }
    }
    close $table3
    exec /bin/chmod ug+w docs/tex/access/table3.txt
    return 0
}
##################################################
####       write_concepts_file()             #####    
##################################################


proc write_concepts_file { } {
    global concepts  ConceptsFile Concepts Routines Processors 
    global sub Comment PETSC_DIR_www files html

    exec /bin/rm -f docs/manualpages/concepts.html
    set concepts_file [ open docs/manualpages/concepts.html w ]

    # Put some  HTML Header 
    puts $concepts_file {<HTML>}
    puts $concepts_file {<TITLE>Concepts_File</TITLE>}
    puts $concepts_file {<BODY>} 
    
    # Put the Table Header
    puts $concepts_file {<H1> Concepts Index </H1>}
    
    # Puts Tabular Header
    puts $concepts_file {<TABLE>}
    puts $concepts_file {<TR HEIGHT=10>}
    puts $concepts_file {<TH WIDTH=4 ><BR></TH>}
    puts $concepts_file {<TH WIDTH=192 ><B><I><FONT SIZE=5>Concepts</FONT></B></I></TH>}
    puts $concepts_file {<TH WIDTH=132 ><B><I><FONT SIZE=5>File Names</FONT></B></I></TH>}
    puts $concepts_file {</TR>}
    puts $concepts_file {</TABLE>}


    foreach concept  $concepts {
        puts $concepts_file {<TABLE>}
        puts $concepts_file {<TD WIDTH=4 ><BR></TD>}
        puts $concepts_file {<TD WIDTH=1000 ><I><FONT SIZE=5>}
        puts $concepts_file $concept
        puts $concepts_file {</FONT></I></TD>}
        puts $concepts_file {</TR>}
        puts $concepts_file {</TABLE>}
        
        foreach subconcept $sub($concept) {
            if { $subconcept != {} } {
                puts $concepts_file {<TABLE>}
                puts $concepts_file {<TD WIDTH=60 ><BR></TD>}
                puts $concepts_file {<TD WIDTH=1000 ><I><FONT SIZE=4>}
                puts $concepts_file $subconcept
                puts $concepts_file {</FONT></I></TD>}
                puts $concepts_file {</TR>}
                puts $concepts_file {</TABLE>}
            }

            set n [ llength $ConceptsFile($concept$subconcept)  ]
            set i 0
            while { $i < $n } {
                set filename [ join [ lindex $ConceptsFile($concept$subconcept) $i ] " " ]
                set temp [ format "<A HREF=\"%s/%s\">%s</A>" $PETSC_DIR_www $filename $filename ]
                puts $concepts_file {<TABLE>}
                puts $concepts_file {<TD WIDTH=192 ><BR></TD>}
                puts $concepts_file {<TD WIDTH=300 >}
                puts $concepts_file $temp
                puts $concepts_file {</TD>}
                puts $concepts_file {</TR>}
                puts $concepts_file {</TABLE>}
                
                set i [ expr $i + 1 ]
            }
        }
    }
 
    # Disclaimer........
    puts $concepts_file "<HR>"
    puts $concepts_file {Note: Not all PETSc examples are currently indexed.
    This list primarily includes examples within the SLES, SNES, TS, and IS components.}

   
    # HTML Tail
    puts $concepts_file {</BODY>} 
    puts $concepts_file {</HTML>}
    
    close $concepts_file
    exec /bin/chmod ug+w docs/manualpages/concepts.html
    return 0
}
##################################################
####       write_routines_file()             #####    
##################################################


proc write_routines_file { } {
    global concepts  ConceptsFile Concepts routines Routines RoutinesFile 
    global Processors Comment PETSC_DIR_www files html
    
    exec /bin/rm -f docs/manualpages/routines.html
    set routines_file [ open docs/manualpages/routines.html w ]

    # Put some  HTML Header 
    puts $routines_file {<HTML>}
    puts $routines_file {<TITLE>Routines_File</TITLE>}
    puts $routines_file {<BODY>} 
    
    # Put the Table Header
    puts $routines_file {<H1> Routines Index </H1>}
    
    # Puts Tabular Header
    puts $routines_file {<TABLE>}
    puts $routines_file {<TR HEIGHT=10>}
    puts $routines_file {<TH WIDTH=4 ><BR></TH>}
    puts $routines_file {<TH WIDTH=192 ><B><I><FONT SIZE=5>Routines</FONT></B></I></TH>}
    puts $routines_file {<TH WIDTH=132 ><B><I><FONT SIZE=5>File Names</FONT></B></I></TH>}
    puts $routines_file {</TR>}
    puts $routines_file {</TABLE>}


    foreach routine  $routines {
        set n [ llength $RoutinesFile($routine)  ]
        puts $routines_file {<TABLE>}
        puts $routines_file {<TD WIDTH=4 ><BR></TD>}
        puts $routines_file {<TD WIDTH=1000 ><I><FONT SIZE=5>}
        puts $routines_file $routine
        puts $routines_file {</FONT></I></TD>}
        puts $routines_file {</TR>}
        puts $routines_file {</TABLE>}
        
        set i 0
        while { $i < $n } {
            set filename [ join [ lindex $RoutinesFile($routine) $i ] " " ]
            set temp [ format "<A HREF=\"%s/%s\">%s</A>" $PETSC_DIR_www $filename $filename ]
            puts $routines_file {<TABLE>}
            puts $routines_file {<TD WIDTH=192 ><BR></TD>}
            puts $routines_file {<TD WIDTH=300 >}
            puts $routines_file $temp
            puts $routines_file {</TD>}
            puts $routines_file {</TR>}
            puts $routines_file {</TABLE>}
            
            set i [ expr $i + 1 ]
        }
    }

    # Disclaimer........
    puts $routines_file "<HR>"
    puts $routines_file {Note: Not all PETSc examples are currently indexed.
    This list primarily includes examples within the SLES, SNES, TS, and IS components.}
    
    # HTML Tail
    puts $routines_file {</BODY>} 
    puts $routines_file {</HTML>}
    
    close $routines_file
    exec /bin/chmod ug+w docs/manualpages/routines.html
    return 0
}
##################################################
####               main()                    #####    
##################################################
# Initialise some global datastructures
# change dir to PETSC_HOME [/home/bsmith/petsc]
proc main { }  {
    global  concepts ConceptsFile Concepts 
    global routines Routines RoutinesFile 
    global Processors Comment PETSC_DIR_www files html
    global sub argc argv env

    set PETSC_HOME $env(PETSC_DIR)
    cd $PETSC_HOME

    # Process the command line arguments
    # Do the brute force way
    set UPDATE_WWW true
    if { $argc == 1  && [lindex $argv 0 ] == "-noupdatewww" } {
        set UPDATE_WWW false
    }
    set PETSC_DIR_www     ../..
    set PETSC_DIR_www_man ../../..
    
    # All the tutorial files containg formated comments
    set files [ glob src/*/examples/{,tutorials}/{*.f,*.F,*.c} src/*/examples/tutorials/umin/{*.F,*.c}]

    # Initailise the data structures
    set concepts {}
    set routines {}
    foreach filename $files {
        set Concepts($filename)   {}
        set Routines($filename)   {}
        set Processors($filename) {}
        set Comment($filename)    ""
    }
    # scan each file and capture the formated comments
    foreach filename $files {  scanfile $filename  }
    
    # For each data entry, eliminate the white spaces in fornt/at the end
    foreach filename $files { 
        deletespace Concepts($filename) 
        deletespace Routines($filename) 
        deletespace Processors($filename)
        deletelinefeed Comment($filename)
    }

    # Do the grouping by Concepts and Routines
    foreach filename $files {
#        foreach concept $Concepts($filename) {
#            lappend ConceptsFile($concept) $filename
#            # add to the concepts list
#            if { [lsearch -exact $concepts $concept] == -1 } {
#                lappend concepts $concept
#            }
#        }
        
        foreach concept $Concepts($filename) {
#            set concept "$concept^TEMP"
            set temp [ split $concept ^ ]
            set size [ llength $temp ]
            set concept [ lindex $temp 0 ]
            set temp [lreplace $temp 0 0]
            if { [lsearch -exact $concepts $concept] == -1 } {
                lappend concepts $concept
            }
            # case when there is no subconcept
            if { $size == 1 } {
                set subconcept {}
                lappend sub($concept)  $subconcept
                lappend ConceptsFile($concept$subconcept) $filename
            } else {
                foreach subconcept $temp {
                    lappend sub($concept)  $subconcept
                    lappend ConceptsFile($concept$subconcept) $filename  
                }
            }
        }

        foreach routine $Routines($filename) {
            lappend RoutinesFile($routine) $filename
            # add to the routines list
            if { [lsearch -exact $routines $routine] == -1 } {
                lappend routines $routine
            }
        }
    }
    set routines [ lsort $routines ]
    set concepts [ lsort $concepts ]
    foreach concept $concepts {
        set sub($concept) [lsort $sub($concept)]
        # make the elements unique
        set temp {}
        foreach subconcept $sub($concept) {
            if { [lsearch -exact $temp $subconcept] == -1 } {
                lappend temp $subconcept
            }
        }
        set sub($concept) $temp
    }

    # Modify the filename and make it hypertext 
    # Just a temporary test.. Must take it away....
    foreach filename $files {
        set tmp [ format "<A HREF=\"%s/%s\">%s</A>" $PETSC_HOME $filename $filename ]
        set html($filename) $tmp
    }
    
    # Print out the data collected in various tables for ms-access 
    #/* write_access_tables  */

    # Write to concepts.html
    write_concepts_file
    write_routines_file

    if { $UPDATE_WWW == "false" } {
        puts "returning early.. not updating wwwmanpages pages."
        return 0
    }
    # Update wwwmanpages
    puts  "updating wwwmanpages pages."
    foreach routine $routines {
        set n [ llength $RoutinesFile($routine)  ]
        set i 0

        set temp [ string first "(" $routine ]
        if { $temp  != -1 } {
            set routine_name [string range  $routine 0 [expr $temp -1 ] ]
        } else {
            set mesg "incorrect Routine: $routine from files  $RoutinesFile($routine)"
            puts stderr $mesg
            return 0
        }
        set routines_file {}
        set temp [ catch { glob docs/manualpages/*/$routine_name.html} routines_file ]
        #puts "processing docs/manualpages/*/$routine_name.html"
        if { $temp != 0 } {
            set mesg "Skipping update of docs/manualpages/*/$routine_name.html. File doesn't exist"
            puts stderr $mesg
            set mesg "$routine_name specified in $RoutinesFile($routine)"
            puts stderr $mesg
            continue
        }

        set routines_fileid [ open $routines_file  r ]
        set routine_file_buff [ read $routines_fileid ]
        close $routines_fileid        
        set buf ""
        while { $i < $n } {
            set filename [ join [ lindex $RoutinesFile($routine) $i ] " " ]
            set temp [ format "<A HREF=\"%s/%s\">%s</A>" $PETSC_DIR_www_man $filename $filename ]
            set buf [format "%s%s%s\n" $buf $temp "<BR>"]
            set i [ expr $i + 1 ]
        }


        # The following code is a bit of a hack, because It is hard
        # to do this properly in tcl ( couldn't figure out an easy
        # way to do sort | uniq with tcl. Here we first check if
        # the Section Examples is already added to the manualpage,
        # and if it is not, then it is added.

        set temp [regexp "<H2>Examples</H2>" $routine_file_buff]
        if { $temp == 0 } {
            # New Section entry for Examples
            set buf [format "%s%s" "<P><H2>Examples</H2>" $buf]
        }
        set buf [format "%s%s" $buf "<P><B>Level:</B>" ]

# This string must match what's in docs/tex/doctext/html.def
        set temp [regsub  "<P><B>Level:</B>" $routine_file_buff $buf routine_file_buff]
        if { $temp == 0 } { 
            puts "******* Unable to modify $routines_file"
        }
        exec /bin/rm -f $routines_file
        set routines_fileid [ open $routines_file w ]
        #puts "Writing to $routines_file"
        puts  $routines_fileid $routine_file_buff
        close $routines_fileid 
        exec /bin/chmod ug+w $routines_file
    }
    return 0
}




#################################################
# the stupid main function is called here    
main
