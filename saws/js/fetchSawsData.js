//modified from PETSc.getAndDisplayDirectory
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

        if (SAWs_prefix == "(null)")//null on first pc I believe (because first pc has no prefix)
            SAWs_prefix = ""; //"(null)" fails populatePcList(), don't know why???

        if (typeof $("#pcList-1"+SAWs_prefix+"text").attr("title") == "undefined" && SAWs_prefix.indexOf("est")==-1) {//it doesn't exist already and doesn't contain 'est'
            $("#o-1").append("<br><b style='margin-left:20px;' title=\"Preconditioner\" id=\"pcList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"pc_type &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList-1"+SAWs_prefix+"\"></select>");
            populatePcList("pcList-1"+SAWs_prefix,SAWs_alternatives,SAWs_pcVal);

            //parse through prefix: first determine what fieldsplit level we are working with, then determine what endtag we are working with

            var fieldsplit="0";

            if(SAWs_prefix.indexOf("fieldsplit_")!=-1) {//still contains at least 1 fieldsplit (assume max of 1 fieldsplit for now...need more test cases)

                var closest=SAWs_prefix.length;//the furthest a keyword could possibly be
                //find index of next keyword (pc, ksp, sub, smoothing, coarse)
                var keywords=["pc","ksp","sub","smoothing","coarse","redundant"];
                var loc="";
                for(var i=0; i<keywords.length; i++) {
                    loc=SAWs_prefix.indexOf(keywords[i]);
                    if(loc < closest && loc != -1)
                        closest=loc;
                }

                var theword = SAWs_prefix.substring(11,closest-1);//omit the first and last underscore

                if(theword != currentFieldsplitWord) {// new fieldsplit
                    currentFieldsplitNumber++;
                    currentFieldsplitWord=theword;
                    var writeLoc = sawsInfo.length;
                    sawsInfo[writeLoc]      = new Object();
                    sawsInfo[writeLoc].data = new Array();
                    sawsInfo[writeLoc].id   = fieldsplit + currentFieldsplitNumber.toString();
                }
                if(currentFieldsplitNumber != -1)
                    fieldsplit = fieldsplit + currentFieldsplitNumber.toString();
            }

            var index = getSawsIndex(fieldsplit);

            var endtag="";
            while(SAWs_prefix!="") {//parse the entire prefix
                var indexFirstUnderscore=SAWs_prefix.indexOf("_");
                var chunk=SAWs_prefix.substring(0,indexFirstUnderscore);//dont include the underscore

                if(chunk=="mg") {//mg_
                    indexFirstUnderscore=SAWs_prefix.indexOf("_",3); //this will actually be the index of the second underscore now since mg prefix has underscore in itself
                    chunk=SAWs_prefix.substring(0,indexFirstUnderscore);//updated chunk
                }

                if(chunk=="mg_levels") {//need to include yet another underscore
                    indexFirstUnderscore=SAWs_prefix.indexOf("_",10); //this will actually be the index of the third underscore
                    chunk=SAWs_prefix.substring(0,indexFirstUnderscore);//updated chunk
                }

                SAWs_prefix=SAWs_prefix.substring(indexFirstUnderscore+1, SAWs_prefix.length);//dont include the underscore

                if(chunk=="mg_coarse" && SAWs_prefix=="")//new mg coarse
                    mgLevelLocation=endtag+"0";

                if(chunk=="ksp" || chunk=="sub" || chunk=="mg_coarse" || chunk=="redundant")
                    endtag+="0";
                else if(chunk.substring(0,10)=="mg_levels_")
                    endtag+=chunk.substring(10,11);//can only be 1 character long for now

                if(chunk.substring(0,10)=="mg_levels_" && SAWs_prefix=="")//new mg levels. it's okay to assume memory was already allocated b/c levels is after coarse
                    sawsInfo[index].data[getSawsDataIndex(index, mgLevelLocation)].mg_levels=parseInt(chunk.substring(10,11))+1;
            }

            if(sawsInfo.length == 0) //special case. manually start off first one
                index = 0;

            if(sawsInfo[index] == undefined) {
                sawsInfo[index]=new Object();
                sawsInfo[index].id = fieldsplit;
            }
            if(sawsInfo[index].data == undefined)//allocate new mem if needed
                sawsInfo[index].data=new Array();
            //search if it has already been created
            if(getSawsDataIndex(index,endtag) == -1) {//doesn't already exist so allocate new memory
                var writeLoc=sawsInfo[index].data.length;
                sawsInfo[index].data[writeLoc]=new Object();
                sawsInfo[index].data[writeLoc].endtag=endtag;
            }
            var index2=getSawsDataIndex(index,endtag);
            sawsInfo[index].data[index2].pc=SAWs_pcVal;

            if (SAWs_pcVal == 'bjacobi') {//some extra data for bjacobi
                //saws does bjacobi_blocks differently than we do. we put bjacoi_blocks as a different endtag than bjacobi dropdown (lower level) but saws puts them on the same level so we need to add a "0" to the endtag
                endtag=endtag+"0";
                if(getSawsDataIndex(index,endtag)==-1) {//need to allocate new memory
                    var writeLoc=sawsInfo[index].data.length;
                    sawsInfo[index].data[writeLoc]=new Object();
                    sawsInfo[index].data[writeLoc].endtag=endtag;
                }
                sawsInfo[index].data[getSawsDataIndex(index,endtag)].bjacobi_blocks = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_bjacobi_blocks"].data[0];
            }

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
            $("#o-1").append("<br><b style='margin-left:20px;' title=\"Krylov method\" id=\"kspList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"ksp_type &nbsp;</b><select class=\"kspLists\" id=\"kspList-1"+SAWs_prefix+"\"></select>");//giving an html element a title creates a tooltip
            populateKspList("kspList-1"+SAWs_prefix,SAWs_alternatives,SAWs_kspVal);

            //parse through prefix...
            //first determine what fieldsplit level we are working with, then determine what endtag we are working with

            var fieldsplit="0";

            if(SAWs_prefix.indexOf("fieldsplit_")!=-1) {//still contains at least 1 fieldsplit (assume max of 1 fieldsplit for now...need more test cases)

                var closest=SAWs_prefix.length;//the furthest a keyword could possibly be
                //find index of next keyword (pc, ksp, sub, smoothing, coarse)
                var keywords=["pc","ksp","sub","smoothing","coarse","redundant"];
                var loc="";
                for(var i=0; i<keywords.length; i++) {
                    loc=SAWs_prefix.indexOf(keywords[i]);
                    if(loc < closest && loc != -1)
                        closest=loc;
                }

                var theword = SAWs_prefix.substring(11,closest-1);//omit the first and last underscore

                if(theword != currentFieldsplitWord) {// new fieldsplit
                    currentFieldsplitNumber++;
                    currentFieldsplitWord=theword;
                    var writeLoc = sawsInfo.length;
                    sawsInfo[writeLoc]      = new Object();
                    sawsInfo[writeLoc].data = new Array();
                    sawsInfo[writeLoc].id   = fieldsplit + currentFieldsplitNumber.toString();
                }
                if(currentFieldsplitNumber != -1)
                    fieldsplit = fieldsplit + currentFieldsplitNumber.toString();
            }

            var index=getSawsIndex(fieldsplit);

            var endtag="";

            while(SAWs_prefix!="") {//parse the entire prefix
                var indexFirstUnderscore=SAWs_prefix.indexOf("_");
                var chunk=SAWs_prefix.substring(0,indexFirstUnderscore);//dont include the underscore

                if(chunk=="mg") {//mg_
                    indexFirstUnderscore=SAWs_prefix.indexOf("_",3); //this will actually be the index of the second underscore now since mg prefix has underscore in itself
                    chunk=SAWs_prefix.substring(0,indexFirstUnderscore);//updated chunk
                }

                if(chunk=="mg_levels") {//need to include yet another underscore
                    indexFirstUnderscore=SAWs_prefix.indexOf("_",10); //this will actually be the index of the third underscore
                    chunk=SAWs_prefix.substring(0,indexFirstUnderscore);//updated chunk
                }

                SAWs_prefix=SAWs_prefix.substring(indexFirstUnderscore+1, SAWs_prefix.length);//dont include the underscore
                if(chunk=="ksp" || chunk=="sub" || chunk=="mg_coarse" || chunk=="redundant")
                    endtag+="0";
                else if(chunk.substring(0,10)=="mg_levels_")
                    endtag+=chunk.substring(10,11);//can only be 1 character long for now
            }

            if(index == -1)
                index = 0;

            if(sawsInfo[index] == undefined) {
                sawsInfo[index]=new Object();
                sawsInfo[index].id = fieldsplit;
            }
            if(sawsInfo[index].data == undefined)//allocate new mem if needed
                sawsInfo[index].data=new Array();
            //search if it has already been created
            if(getSawsDataIndex(index,endtag) == -1) {//doesn't already exist so allocate new memory
                var writeLoc=sawsInfo[index].data.length;
                sawsInfo[index].data[writeLoc]=new Object();
                sawsInfo[index].data[writeLoc].endtag=endtag;
            }
            var index2=getSawsDataIndex(index,endtag);
            sawsInfo[index].data[index2].ksp=SAWs_kspVal;
        }
    }

    SAWs.displayDirectoryRecursive(sub.directories,divEntry,0,"");//this function is not in SAWs API ?
}