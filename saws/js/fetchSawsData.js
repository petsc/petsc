//modified from PETSc.getAndDisplayDirectory
SAWsGetAndDisplayDirectory = function(names,divEntry){
    jQuery(divEntry).html(""); //clears divEntry
    SAWs.getDirectory(names,SAWsDisplayDirectory,divEntry);
}

