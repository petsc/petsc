#! /usr/local/tcl/bin/tclsh


##################################################
####               main()                    #####    
##################################################
proc main { }  {
    set PETSC_DIR /home/bsmith/petsc
    cd $PETSC_DIR
    
    set files [ glob docs/www/man*/*.html ]
    
    foreach filename $files {
        # open the file
        set fileid [ open $filename r ]
        set filebuff [ read $fileid ]
        
        #The following string is from examplesindex.tcl which inserts in these files
        set start_data [ string first "<H2>Examples</H2>"  $filebuff ]
        if { $start_data == -1 } {
            set mesg [ file tail $filename ]
            regsub -all ".html" $mesg "()" mesg
            puts stdout $mesg
        }
        
        close $fileid
    }
}


#################################################
# the stupid main function is called here    
main
