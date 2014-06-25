//GetAndDisplayDirectory: modified from PETSc.getAndDisplayDirectory
//------------------------------------------
SAWsGetAndDisplayDirectory = function(names,divEntry){
    jQuery(divEntry).html(""); //clears divEntry
    SAWs.getDirectory(names,SAWsDisplayDirectory,divEntry);
}


//DisplayDirectory: modified from PETSc.displayDirectory
//------------------------------------------------------
SAWsDisplayDirectory = function(sub,divEntry)
{
    globaldirectory[divEntry] = sub;

    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {//this function is nearly always called
        SAWs.updateDirectoryFromDisplay(divEntry);
        sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
        SAWs.postDirectory(sub);
        jQuery(divEntry).html("");//empty divEntry
        window.setTimeout(SAWsGetAndDisplayDirectory,1000,null,divEntry);//calls SAWsGetAndDisplayDirectory(null, divEntry) after 1000ms
    }

    recordSawsData(sub);

    SAWs.displayDirectoryRecursive(sub.directories,divEntry,0,"");
}

function recordSawsData(sub) {

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

        //if (typeof $("#pcList-1"+SAWs_prefix+"text").attr("title") == "undefined" && SAWs_prefix.indexOf("est") == -1) {//it doesn't exist already and doesn't contain 'est'
            //$("#o-1").append("<div id='saws"+serverOptionsCounter+"'><b style='margin-left:20px;' title=\"Preconditioner\" id=\"pcList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"pc_type &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList-1"+SAWs_prefix+"\"></select></div>");
            //serverOptionsCounter++; //DO NOT DELETE THIS CHUNK OF CODE
            //populatePcList("pcList-1"+SAWs_prefix,SAWs_alternatives,SAWs_pcVal);

            //parse through prefix
            var endtag   = parsePrefix(SAWs_prefix).endtag;//this func returns an object with two pieces of information: the endtag and the new word encountered (if any)
            var newWord  = parsePrefix(SAWs_prefix).newWord;

            var writeLoc = getSawsIndex(endtag);

            if(writeLoc == -1) {//need to alloc memory
                writeLoc                    = sawsInfo.length;
                sawsInfo[writeLoc]          = new Object();
                sawsInfo[writeLoc].endtag   = endtag;
            }

            sawsInfo[writeLoc].pc              = SAWs_pcVal;
            sawsInfo[writeLoc].pc_alternatives = SAWs_alternatives.slice();//deep copy of alternatives

            if (SAWs_pcVal == 'bjacobi') {//some extra data for pc=bjacobi
                sawsInfo[writeLoc].bjacobi_blocks = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_bjacobi_blocks"].data[0];
            }

            if(SAWs_pcVal == 'mg') {//some extra data for pc=multigrid
                sawsInfo[writeLoc].mg_levels = 1;//make it 1 level by default. when another mg_level is encountered, this variable will be overwritten
            }

            if(SAWs_pcVal == 'fieldsplit') { //some extra data for pc=fieldsplit
                sawsInfo[writeLoc].fieldsplit_blocks = 1;//make it 1 block by default. when another fieldsplit is encountered, this variable will be overwritten
            }

            //check if parent was mg because then this child is a mg_level and we might need to record a new record for mg_level. we need to do this because the variable mg_levels is still not available in saws yet.
            var parentEndtag = getParent(endtag);
            var parentIndex  = getSawsIndex(parentEndtag);

            if(parentIndex != -1 && sawsInfo[parentIndex].pc == "mg") { //check to see if parent was mg
                var currentLevel                = endtag.substring(endtag.lastIndexOf('_')+1, endtag.length);//everything after the last underscore
                currentLevel                    = parseInt(currentLevel);
                sawsInfo[parentIndex].mg_levels = currentLevel+1;//if we are on level 0 then that means there was 1 level so far
            }

            if(parentIndex != -1 && sawsInfo[parentIndex].pc == "fieldsplit"){ //cheeck to see if parent was fieldsplit
                var currentLevel                        = endtag.substring(endtag.lastIndexOf('_')+1, endtag.length);//everything after the last underscore
                currentLevel                            = parseInt(currentLevel);
                sawsInfo[parentIndex].fieldsplit_blocks = currentLevel + 1;
                sawsInfo[writeLoc].name                 = newWord;//important! record name of the fieldsplit
            }
        //}

    }

    /* ---------------------------------- KSP OPTIONS --------------------------------- */

    else if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Krylov Method (KSP) options") {
        var SAWs_kspVal       = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].alternatives;
        var SAWs_prefix       = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.prefix.data[0];

        if (SAWs_prefix == "(null)")
            SAWs_prefix = "";

        //if (typeof $("#kspList-1"+SAWs_prefix+"text").attr("title") == "undefined" && SAWs_prefix.indexOf("est")==-1) {//it doesn't exist already and doesn't contain 'est'
            //$("#o-1").append("<div id='saws"+serverOptionsCounter+"'><b style='margin-left:20px;' title=\"Krylov method\" id=\"kspList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"ksp_type &nbsp;</b><select class=\"kspLists\" id=\"kspList-1"+SAWs_prefix+"\"></select></div>");//giving an html element a title creates a tooltip
            //serverOptionsCounter++; //DO NOT DELETE THIS CHUNK OF CODE
            //populateKspList("kspList-1"+SAWs_prefix,SAWs_alternatives,SAWs_kspVal);

            //parse through prefix
            var endtag   = parsePrefix(SAWs_prefix).endtag;//this func returns an object with two pieces of information: the endtag and the new word encountered (if any)
            var newWord  = parsePrefix(SAWs_prefix).newWord;

            var writeLoc = getSawsIndex(endtag);

            if(writeLoc == -1) {//need to alloc memory
                writeLoc                    = sawsInfo.length;
                sawsInfo[writeLoc]          = new Object();
                sawsInfo[writeLoc].endtag   = endtag;
            }

            sawsInfo[writeLoc].ksp              = SAWs_kspVal;
            sawsInfo[writeLoc].ksp_alternatives = SAWs_alternatives.slice();//deep copy of alternatives

        //}
    }
}