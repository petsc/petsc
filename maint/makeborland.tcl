#!/usr/bin/env tclsh
# $Id: makecpp.tcl,v 1.18 1999/02/03 23:12:53 balay Exp $ 

proc updatecommon { dir } {
    if [catch { cd $dir } err ] {
        puts stderr $err
    }
    puts "************ processing bmake/common* files"
    set files [ glob -nocomplain bmake/common* ]
    foreach filename $files {
        updatemakefile  $filename
    }

}

proc movefilesin { dir } {
    if [catch { cd $dir } err ] {
        puts stderr $err
    }
    set mesg "############ $dir"
    puts $mesg

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
        # Change 'sles.o' to 'sles.obj'
        #
        regsub -all "\\\.o "  $databuff "\.obj " databuff
        regsub -all "\\\.o\n"  $databuff "\.obj\n" databuff
        regsub -all "\\\.o\\\\"  $databuff "\.obj\\" databuff
        regsub -all "\\\.o:"  $databuff "\.obj:" databuff
        #
        # Change / to \ 
        #
        #regsub -all "/" $databuff "\\" databuff
        #
        # Add .obj to siffixes:
        #
        regsub -all "SUFFIXES:" $databuff "SUFFIXES: \.obj" databuff
        #
        # Strip out the LINKER option -o ex* from the makefiles
        #
        regsub -all {\-o[ ]+e[^ ]* } $databuff "" databuff
        #
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
    puts "makeborland.tcl: please specify the PETSc dir"
    return
}
puts "**** make cpp in $PETSC_HOME ************"
### Update the makefiles ##########
movefilesin $PETSC_HOME
### Now update the bmake/common* files #########
updatecommon $PETSC_HOME
