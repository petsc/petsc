#! /usr/local/tcl/bin/tclsh
#!/gnuwin32/b18/tcl/bin/tclsh76.exe
# $Id: htmlkeywords.tcl,v 1.6 1997/09/10 19:44:00 balay Exp $ 


proc movefilesin { dir } {
    if [catch { cd $dir } err ] {
        puts stderr $err
    }
    set mesg "############ $dir"
    puts $mesg

#    set files [ glob -nocomplain *.c ]
#    foreach filename $files {
#        set fileroot [file rootname $filename]
#        set mesg "mv $fileroot.c $fileroot.cpp"
#        puts $mesg
#        exec /bin/mv $fileroot.c $fileroot.cpp
#    }

    # Process the makefile
#    puts "Updating makefile "
    updatemakefile  "makefile"

    # Process the subdirs
    set files [ glob -nocomplain * ]
    foreach filename $files {
        if [ file isdirectory $filename ] {
            movefilesin $dir/$filename
        }
        cd $dir
    }
    return 0
}


proc updatemakefile { makefile } {
    if [ file exists $makefile ] {
        set fileid [ open $makefile r ]
        set databuff [ read $fileid ]
        close $fileid

        #
        # Change 'include' to '!include'
        #
        regsub  -all  "\ninclude" $databuff  "\n!include" databuff
        #
        # Change -D__SDIR__='"$(LOCDIR)"' to /D__SDIR__="\"$(LOCDIR)\""
        #
        regsub -all  "\\\-D" $databuff "/D" databuff
        regsub -all "\'\""  $databuff "\"\\\"" databuff
        regsub -all "\"\'"  $databuff "\\\"\"" databuff
        #
        # Change 'sles.o' to 'sles.obj'
        #
        regsub -all "\\\.o "  $databuff "\.obj " databuff
        regsub -all "\\\.o\n"  $databuff "\.obj\n" databuff
        regsub -all "\\\.o\\\\"  $databuff "\.obj\\" databuff
        #
        # Change / to \ 
        #
#       regsub -all "/" $databuff "\\" databuff
        #
        # Strip out the LINKER option -o ex* from the makefiles
        #
        regsub -all {\-o[ ]+e[^ ]* } $databuff "" databuff
        #
        # Strip out -f option to RM from the makefile.
        #
        regsub -all { \-f} $databuff "" databuff        

        set fileid [ open $makefile w ]
        puts $fileid $databuff
        close $fileid
    }
}

proc copy { a b } {
    set fileid [ open $a r ]
    set databuff [ read $fileid ]
    close $fileid
    set fileid [ open $b w ]
    puts $fileid $databuff
    close $fileid
}
 

#set PETSC_HOME $env(PETSC_DIR)  
if { $argc == 1 } {
    set PETSC_HOME [ lindex $argv 0 ] 
} else {
    puts "makecpp.tcl: please specify the PETSc dir"
    return
}
puts "**** make cpp in $PETSC_HOME ************"
movefilesin $PETSC_HOME
#movefilesin c:/work/petsc/xyz
copy $PETSC_HOME/bmake/nt/makefile.dos $PETSC_HOME/makefile
