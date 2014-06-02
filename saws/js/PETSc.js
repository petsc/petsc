
PETSc = {}

PETSc.getAndDisplayDirectory = function(names,divEntry){
    window.location = 'pcoptions.html'

    //below are skipped now ...
    jQuery(divEntry).html("")
    SAWs.getDirectory(names,PETSc.displayDirectory,divEntry)
}

PETSc.displayDirectory = function(sub,divEntry)
{
    globaldirectory[divEntry] = sub
    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {
        jQuery(divEntry).append("<center><input type=\"button\" value=\"Continue\" id=\"continue\"></center>")
        jQuery('#continue').on('click', function(){
            SAWs.updateDirectoryFromDisplay(divEntry)
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
            SAWs.postDirectory(sub);
            jQuery(divEntry).html("");
            window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divEntry);
        })
    }
    //var SAWs_pcVal = JSON.stringify(sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0]);
    //alert("SAWs_pcVal="+SAWs_pcVal);
    //alert(JSON.stringify(sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].alternatives)) //pcList

    //alert(JSON.stringify(sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables))

    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
        window.location = 'pcoptions.html'
    } else {
        SAWs.displayDirectoryRecursive(sub.directories,divEntry,0,"")
    }
}