//this function uses parsePrefix2.js to record the JSON data in sub into the array 'data'

function recordSawsData(data, sub) {

    if(sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories == undefined) {
        alert("Error! Most likely cause: invalid options.");
        return;
    }

    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
        var SAWs_pcVal = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].alternatives;
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["prefix"].data[0];

        if (SAWs_prefix == "(null)")
            SAWs_prefix = "";

        //parse through prefix
        var endtag   = parsePrefix(data, SAWs_prefix).endtag;//this func returns an object with two pieces of information: the endtag and the new word encountered (if any)
        var newWord  = parsePrefix(data, SAWs_prefix).newWord;

        var writeLoc = getIndex(data,endtag);

        if(writeLoc == -1) {//need to alloc memory
            writeLoc                = data.length;
            data[writeLoc]          = new Object();
            data[writeLoc].endtag   = endtag;
        }

        data[writeLoc].pc              = SAWs_pcVal;
        data[writeLoc].pc_alternatives = SAWs_alternatives.slice();//deep copy of alternatives

        if (SAWs_pcVal == 'bjacobi') {//some extra data for pc=bjacobi
            data[writeLoc].bjacobi_blocks = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_bjacobi_blocks"].data[0];
        }

        if(SAWs_pcVal == 'mg') {//some extra data for pc=multigrid
            data[writeLoc].mg_levels = 1;//make it 1 level by default. when another mg_level is encountered, this variable will be overwritten
        }

        if(SAWs_pcVal == 'fieldsplit') { //some extra data for pc=fieldsplit
            data[writeLoc].fieldsplit_blocks = 1;//make it 1 block by default. when another fieldsplit is encountered, this variable will be overwritten
        }

        //check if parent was mg because then this child is a mg_level and we might need to record a new record for mg_level. we need to do this because the variable mg_levels is still not available in saws yet.
        var parentEndtag = getParent(endtag);
        var parentIndex  = getIndex(data,parentEndtag);

        if(parentIndex != -1 && data[parentIndex].pc == "mg") { //check to see if parent was mg
            var currentLevel                = endtag.substring(endtag.lastIndexOf('_')+1, endtag.length);//everything after the last underscore
            currentLevel                    = parseInt(currentLevel);
            data[parentIndex].mg_levels     = currentLevel+1;//if we are on level 0 then that means there was 1 level so far
        }

        if(parentIndex != -1 && data[parentIndex].pc == "fieldsplit"){ //cheeck to see if parent was fieldsplit
            var currentLevel                        = endtag.substring(endtag.lastIndexOf('_')+1, endtag.length);//everything after the last underscore
            currentLevel                            = parseInt(currentLevel);
            data[parentIndex].fieldsplit_blocks     = currentLevel + 1;
            if(newWord != "")
                data[writeLoc].name                 = newWord;//important! record name of the fieldsplit
        }
    }

    /* ---------------------------------- KSP OPTIONS --------------------------------- */

    else if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Krylov Method (KSP) options") {
        var SAWs_kspVal       = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].alternatives;
        var SAWs_prefix       = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.prefix.data[0];

        if (SAWs_prefix == "(null)")
            SAWs_prefix = "";

        //parse through prefix
        var endtag   = parsePrefix(data, SAWs_prefix).endtag;//this func returns an object with two pieces of information: the endtag and the new word encountered (if any)
        var newWord  = parsePrefix(data, SAWs_prefix).newWord;

        var writeLoc = getIndex(data,endtag);

        if(writeLoc == -1) {//need to alloc memory
            writeLoc                = data.length;
            data[writeLoc]          = new Object();
            data[writeLoc].endtag   = endtag;
        }

        data[writeLoc].ksp              = SAWs_kspVal;
        data[writeLoc].ksp_alternatives = SAWs_alternatives.slice();//deep copy of alternatives
    }
}