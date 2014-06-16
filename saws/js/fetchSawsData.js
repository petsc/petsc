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
    //alert("2. DisplayDirectory: sub="+sub+"; divEntry="+divEntry);
    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {//this function is nearly always called
        SAWs.updateDirectoryFromDisplay(divEntry);
        sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
        SAWs.postDirectory(sub);
        jQuery(divEntry).html("");//empty divEntry
        window.setTimeout(SAWsGetAndDisplayDirectory,500,null,divEntry);//calls SAWsGetAndDisplayDirectory(null, divEntry) after 500ms
    }

    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
        var SAWs_pcVal = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].alternatives;
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["prefix"].data[0];

        if (SAWs_prefix == "(null)")//null on first pc
            SAWs_prefix = ""; //"(null)" fails populatePcList()

        if (typeof $("#pcList-1"+SAWs_prefix+"text").attr("title") == "undefined" && SAWs_prefix.indexOf("est") == -1) {//it doesn't exist already and doesn't contain 'est'
            $("#o-1").append("<div id='saws"+serverOptionsCounter+"'><b style='margin-left:20px;' title=\"Preconditioner\" id=\"pcList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"pc_type &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList-1"+SAWs_prefix+"\"></select></div>");
            serverOptionsCounter++;
            populatePcList("pcList-1"+SAWs_prefix,SAWs_alternatives,SAWs_pcVal);

            //parse through prefix
            var fieldsplit = parsePrefixForFieldsplit(SAWs_prefix);
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
                //saws does bjacobi_blocks differently than we do. we put bjacoi_blocks as a different endtag than bjacobi dropdown (lower level) but saws puts them on the same level so we need to add a "0" to the endtag
                endtag = endtag + "0";
                if (getSawsDataIndex(index,endtag) == -1) {//need to allocate new memory
                    var writeLoc = sawsInfo[index].data.length;
                    sawsInfo[index].data[writeLoc]        = new Object();
                    sawsInfo[index].data[writeLoc].endtag = endtag;
                }
                sawsInfo[index].data[getSawsDataIndex(index,endtag)].bjacobi_blocks = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_bjacobi_blocks"].data[0];
            }

            /*if(SAWs_pcVal == 'mg') {//some extra data for pc=multigrid
                //petsc does mg_levels differently as well. so we need to allocate memory for it.
                endtag = endtag + "0";
                
            } ignore this code for now. I'm working on it.

            //lastly, check if parent was mg because then this child is a mg_level and we might need to record a new record for mg_level
            var parentEndtag = endtag.substring
            if(sawsInfo[index].data[getSawsDataIndex(index, endtag.substring*/

        }

    }

    /* ---------------------------------- KSP OPTIONS --------------------------------- */

    else if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Krylov Method (KSP) options") {
        var SAWs_kspVal       = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].alternatives;
        var SAWs_prefix       = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.prefix.data[0];

        if (SAWs_prefix == "(null)")
            SAWs_prefix = "";

        if (typeof $("#kspList-1"+SAWs_prefix+"text").attr("title") == "undefined" && SAWs_prefix.indexOf("est")==-1) {//it doesn't exist already and doesn't contain 'est'
            $("#o-1").append("<div id='saws"+serverOptionsCounter+"'><b style='margin-left:20px;' title=\"Krylov method\" id=\"kspList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"ksp_type &nbsp;</b><select class=\"kspLists\" id=\"kspList-1"+SAWs_prefix+"\"></select></div>");//giving an html element a title creates a tooltip
            serverOptionsCounter++;
            populateKspList("kspList-1"+SAWs_prefix,SAWs_alternatives,SAWs_kspVal);

            //parse through prefix...
            var fieldsplit = parsePrefixForFieldsplit(SAWs_prefix);
            var index      = getSawsIndex(fieldsplit);
            var endtag     = parsePrefixForEndtag(SAWs_prefix, index);

            if (index == -1)
                index = 0;

            allocateMemory(fieldsplit, endtag, index);

            var index2 = getSawsDataIndex(index,endtag);
            sawsInfo[index].data[index2].ksp = SAWs_kspVal;
            sawsInfo[index].data[index2].ksp_alternatives = SAWs_alternatives.slice();//deep copy
        }
    }

    SAWs.displayDirectoryRecursive(sub.directories,divEntry,0,"");//this function is not in SAWs API ?
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