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

        if (SAWs_prefix == "(null)")//null on first pc
            SAWs_prefix = ""; //"(null)" fails populatePcList()

        //if (typeof $("#pcList-1"+SAWs_prefix+"text").attr("title") == "undefined" && SAWs_prefix.indexOf("est") == -1) {//it doesn't exist already and doesn't contain 'est'
            //$("#o-1").append("<div id='saws"+serverOptionsCounter+"'><b style='margin-left:20px;' title=\"Preconditioner\" id=\"pcList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"pc_type &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList-1"+SAWs_prefix+"\"></select></div>");
            //serverOptionsCounter++; //DO NOT DELETE THIS CHUNK OF CODE
            //populatePcList("pcList-1"+SAWs_prefix,SAWs_alternatives,SAWs_pcVal);

            //parse through prefix
            var fieldsplit = parsePrefixForFieldsplit(SAWs_prefix).fieldsplit;//this func returns an object with two pieces of information: the fieldsplit id and the new word encountered (if any)
            var newWord    = parsePrefixForFieldsplit(SAWs_prefix).newWord;

            if(getSawsIndex(fieldsplit) == -1) {//need to alloc memory for the new fieldsplit
                var writeLoc = sawsInfo.length;
                sawsInfo[writeLoc]      = new Object();
                sawsInfo[writeLoc].data = new Array();
                sawsInfo[writeLoc].id   = fieldsplit;
                sawsInfo[writeLoc].name = newWord;//record fieldsplit name in sawsInfo[]
            }

            var index      = getSawsIndex(fieldsplit);
            var endtag     = parsePrefixForEndtag(SAWs_prefix, index);

            if (sawsInfo.length == 0) {//special case. manually start off first one
                index = 0;
            }

            allocateMemory(fieldsplit, endtag, index);

            var index2 = getSawsDataIndex(index,endtag);
            sawsInfo[index].data[index2].pc              = SAWs_pcVal;
            sawsInfo[index].data[index2].pc_alternatives = SAWs_alternatives.slice();//deep copy of alternatives

            if (SAWs_pcVal == 'bjacobi') {//some extra data for bjacobi
                //petsc does bjacobi_blocks differently than we do. we put bjacoi_blocks as a different endtag than bjacobi dropdown (lower level) but saws puts them on the same level so we need to add a "0" to the endtag
                endtag = endtag + "0";
                if (getSawsDataIndex(index,endtag) == -1) {//need to allocate new memory
                    var writeLoc                          = sawsInfo[index].data.length;
                    sawsInfo[index].data[writeLoc]        = new Object();
                    sawsInfo[index].data[writeLoc].endtag = endtag;
                }
                sawsInfo[index].data[getSawsDataIndex(index,endtag)].bjacobi_blocks = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_bjacobi_blocks"].data[0];
            }

            if(SAWs_pcVal == 'mg') {//some extra data for pc=multigrid. petsc does mg_levels differently (see above explanation for bjacobi)
                endtag = endtag + "0";

                if(getSawsDataIndex(index,endtag) == -1) {//need to allocate new memory
                    var writeLoc                          = sawsInfo[index].data.length;
                    sawsInfo[index].data[writeLoc]        = new Object();
                    sawsInfo[index].data[writeLoc].endtag = endtag;
                }
                sawsInfo[index].data[getSawsDataIndex(index,endtag)].mg_levels = 1;//make it 1 level by default until another mg_level is encountered in which this variable will be overwritten
            }

            //lastly, check if parent was mg because then this child is a mg_level and we might need to record a new record for mg_level. we need to do this because the variable mg_levels is still not available in saws yet.
            //using a global mg variable doesn't allow for nested mg
            var parentEndtag = endtag.substring(0,endtag.length-1);//knock off the last character
            if(sawsInfo[index].data[getSawsDataIndex(index, parentEndtag)].pc == "mg") {
                var mg_levels_endtag = parentEndtag + "0";
                var currentLevel = endtag.charAt(endtag.length-1); //the last character
                currentLevel = parseInt(currentLevel);
                sawsInfo[index].data[getSawsDataIndex(index, mg_levels_endtag)].mg_levels = currentLevel+1;//if we are on level 0 then that means there was 1 level so far
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

            //parse through prefix...
            var fieldsplit = parsePrefixForFieldsplit(SAWs_prefix).fieldsplit;
            var index      = getSawsIndex(fieldsplit);
            var endtag     = parsePrefixForEndtag(SAWs_prefix, index);

            if (index == -1)
                index = 0;

            allocateMemory(fieldsplit, endtag, index);

            var index2 = getSawsDataIndex(index,endtag);
            sawsInfo[index].data[index2].ksp = SAWs_kspVal;
            sawsInfo[index].data[index2].ksp_alternatives = SAWs_alternatives.slice();//deep copy
        //}
    }
}


//this function allocates memory in sawsInfo if needed to fit the new data from SAWs
function allocateMemory(fieldsplit, endtag, index) {

    if (sawsInfo[index] == undefined) {
        sawsInfo[index]    = new Object();
        sawsInfo[index].id = fieldsplit;
    }
    if (sawsInfo[index].data == undefined) {//allocate new mem if needed
        sawsInfo[index].data = new Array();
    }

    //search if it has already been created
    if (getSawsDataIndex(index,endtag) == -1) {//doesn't already exist so allocate new memory
        var writeLoc = sawsInfo[index].data.length;
        sawsInfo[index].data[writeLoc]        = new Object();
        sawsInfo[index].data[writeLoc].endtag = endtag;
    }
}