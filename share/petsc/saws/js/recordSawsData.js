//this function uses parsePrefix.js to record the JSON data in sub into the object 'data'

function recordSawsData(data, sub) {

    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
        var SAWs_pcVal = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].alternatives;
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["prefix"].data[0];

        if (SAWs_prefix == "(null)")
            SAWs_prefix = "";

        //this func returns an object with two pieces of information: the endtag and the new word encountered (if any)
        var parsedInfo = parsePrefix(data, SAWs_prefix);
        var endtag     = parsedInfo.endtag;
        var newWord    = parsedInfo.newWord;

        if(data[endtag] == undefined)
            data[endtag] = new Object();

        data[endtag].pc_type              = SAWs_pcVal;
        data[endtag].pc_type_alternatives = SAWs_alternatives.slice(); //deep copy of alternatives

        if (SAWs_pcVal == 'bjacobi') {//some extra data for pc=bjacobi
            data[endtag].pc_bjacobi_blocks = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_bjacobi_blocks"].data[0];
        }

        if(SAWs_pcVal == 'mg') {//some extra data for pc=multigrid
            data[endtag].pc_mg_levels = 1;//make it 1 level by default. when another mg_level is encountered, this variable will be overwritten
        }

        if(SAWs_pcVal == 'fieldsplit') { //some extra data for pc=fieldsplit
            data[endtag].pc_fieldsplit_blocks = 1;//make it 1 block by default. when another fieldsplit is encountered, this variable will be overwritten
        }

        //check if parent was mg because then this child is a mg_level and we might need to record a new record for mg_level. we need to do this because the variable mg_levels is still not available in saws yet.
        var parentEndtag = getParent(endtag);

        if(data[parentEndtag] != undefined && data[parentEndtag].pc_type == "mg") { //check to see if parent was mg
            var currentLevel                = endtag.substring(endtag.lastIndexOf('_')+1, endtag.length);//everything after the last underscore
            currentLevel                    = parseInt(currentLevel);
            data[parentEndtag].pc_mg_levels = currentLevel+1;//if we are on level 0 then that means there was 1 level so far
        }

        if(data[parentEndtag] != undefined && data[parentEndtag].pc_type == "fieldsplit"){ //cheeck to see if parent was fieldsplit
            var currentLevel                        = endtag.substring(endtag.lastIndexOf('_')+1, endtag.length);//everything after the last underscore
            currentLevel                            = parseInt(currentLevel);
            data[parentEndtag].pc_fieldsplit_blocks = currentLevel + 1;
            if(newWord != "")
                data[endtag].name                   = newWord;//important! record name of the fieldsplit
        }
    }

    /* ---------------------------------- KSP OPTIONS --------------------------------- */

    else if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Krylov Method (KSP) options") {
        var SAWs_kspVal       = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].alternatives;
        var SAWs_prefix       = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.prefix.data[0];

        if (SAWs_prefix == "(null)")
            SAWs_prefix = "";

        //this func returns an object with two pieces of information: the endtag and the new word encountered (if any)
        var parsedInfo = parsePrefix(data, SAWs_prefix);
        var endtag     = parsedInfo.endtag;
        var newWord    = parsedInfo.newWord;

        data[endtag].ksp_type              = SAWs_kspVal;
        data[endtag].ksp_type_alternatives = SAWs_alternatives.slice();//deep copy of alternatives
    }
}
