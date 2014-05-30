/*
  pcGetDetailedInfo - get detailed information from the pclists
  input:
    pcListID
    prefix - prefix of the options in the solverTree
    recursionCounter - id of A-div we are currently working in
  output:
    matInfo.string
    matInfo.stringshort
*/
function pcGetDetailedInfo(pcListID, prefix,recursionCounter,matInfo)
{
    var pcSelectedValue=$("#"+pcListID).val();
    var info      = "";
    var infoshort = "";
    var endtag,myendtag;

    if (pcListID.indexOf("_") == -1) {//dealing with original pcList in the oDiv (not generated for suboptions)
	endtag = "_"; // o-th solver level
    } else {
        var loc = pcListID.indexOf("_");
        endtag = pcListID.substring(loc); // endtag of input pcListID, eg. _, _0, _00, _10
    }
    //alert("pcGetDetailedInfo: pcListID="+pcListID+"; pcSelectedValue= "+pcSelectedValue+"; endtag= "+endtag);

    switch(pcSelectedValue) {
    case "mg" :
        myendtag = endtag+"0";
	var mgType       = $("#mgList" + recursionCounter + myendtag).val();
        var mgLevels     = $("#mglevels" + recursionCounter + myendtag).val();
	info   += "<br />"+prefix+"pc_mg_type " + mgType + "<br />"+prefix+"pc_mg_levels " + mgLevels;
        prefix += "mg_";
        var index=getMatIndex(recursionCounter);
        matInfo[index].string      += info;
        matInfo[index].stringshort += infoshort;

        var smoothingKSP;
        var smoothingPC;
        var coarseKSP;
        var coarsePC;
        var level;

        //is a composite pc so there will be a div in the next position
        var generatedDiv="";
        generatedDiv = $("#"+pcListID).next().get(0).id; //this will be a div
        //level = mgLevels-1;
        level = 0;
        for (var i=0; i<$("#"+generatedDiv).children().length; i++) { //loop over all pcLists under this Div
	    var childID = $("#"+generatedDiv).children().get(i).id;
	    if ($("#"+childID).is(".pcLists")) {//has more pc lists that need to be taken care of recursively
                info      = "";
                infoshort = "";
                if (level != 0) {
		    if(level<10)//still using numbers
			myendtag = endtag+level;
		    else
			myendtag = endtag+'abcdefghijklmnopqrstuvwxyz'.charAt(level-10);//add the correct char

                    smoothingKSP = $("#kspList" + recursionCounter + myendtag).val();
	            smoothingPC  = $("#pcList" + recursionCounter + myendtag).val();
                    info   += "<br />"+prefix+"levels_"+level+"_ksp_type "+smoothingKSP+"<br />"+prefix+"levels_"+level+"_pc_type "+smoothingPC;
                    infoshort += "<br />Level "+level+" -- KSP: "+smoothingKSP+"; PC: "+smoothingPC;
                    var myprefix = prefix+"levels_"+level+"_";
                } else if (level == 0) {
                    myendtag = endtag+"0";
                    coarseKSP    = $("#kspList" + recursionCounter + myendtag).val();
	            coarsePC     = $("#pcList" + recursionCounter + myendtag).val();
                    info   += "<br />"+prefix+"coarse_ksp_type "+coarseKSP+"<br />"+prefix+"coarse_pc_type "+coarsePC;
                    infoshort += "<br />Coarse Grid -- KSP: "+coarseKSP+"; PC: "+coarsePC;
                    var myprefix = prefix+"coarse_";
                } else {
                    alert("Error: mg level cannot be "+level);
                }
                var index=getMatIndex(recursionCounter);
                matInfo[index].string      += info;
                matInfo[index].stringshort += infoshort;

                pcGetDetailedInfo(childID,myprefix,recursionCounter,matInfo);
                level++;
	    }
        }
        return "";
        break;

    case "redundant" :
        endtag += "0"; // move to next solver level
	var redundantNumber = $("#redundantNumber" + recursionCounter + endtag).val();
	var redundantKSP    = $("#kspList" + recursionCounter + endtag).val();
	var redundantPC     = $("#pcList" + recursionCounter + endtag).val();

        info += "<br />"+prefix+"pc_redundant_number " + redundantNumber;
        prefix += "redundant_";
	info += "<br />"+prefix+"ksp_type " + redundantKSP +"<br />"+prefix+"pc_type " + redundantPC;
        infoshort += "<br />PCredundant -- KSP: " + redundantKSP +"; PC: " + redundantPC;
        break;

    case "bjacobi" :
        endtag += "0"; // move to next solver level
	var bjacobiBlocks = $("#bjacobiBlocks" + recursionCounter + endtag).val();
	var bjacobiKSP    = $("#kspList" + recursionCounter + endtag).val();
	var bjacobiPC     = $("#pcList" + recursionCounter + endtag).val();
        info   += "<br />"+prefix+"pc_bjacobi_blocks "+bjacobiBlocks; // option for previous solver level
        prefix += "sub_";
        info   += "<br />"+prefix+"ksp_type "+bjacobiKSP+"<br />"+prefix+"pc_type "+bjacobiPC;
        infoshort  += "<br /> PCbjacobi -- KSP: "+bjacobiKSP+"; PC: "+bjacobiPC;
        break;

    case "asm" :
        endtag += "0"; // move to next solver level
        var asmBlocks  = $("#asmBlocks" + recursionCounter + endtag).val();
        var asmOverlap = $("#asmOverlap" + recursionCounter + endtag).val();
	var asmKSP     = $("#kspList" + recursionCounter + endtag).val();
	var asmPC      = $("#pcList" + recursionCounter + endtag).val();
	info   +=  "<br />"+prefix+"pc_asm_blocks  " + asmBlocks + " "+prefix+"pc_asm_overlap "+ asmOverlap;
        prefix += "sub_";
        info   += "<br />"+prefix+"ksp_type " + asmKSP +" "+prefix+"pc_type " + asmPC;
        infoshort += "<br />PCasm -- KSP: " + asmKSP +"; PC: " + asmPC;
        break;

    case "ksp" :
        endtag += "0"; // move to next solver level
	var kspKSP = $("#kspList" + recursionCounter + endtag).val();
	var kspPC  = $("#pcList" + recursionCounter + endtag).val();
        prefix += "ksp_";
        info   += "<br />"+prefix+"ksp_type " + kspKSP + " "+prefix+"pc_type " + kspPC;
        infoshort += "<br />PCksp -- KSP: " + kspKSP + "; PC: " + kspPC;
        break;

    default :
    }

    if  (info.length == 0) return ""; //is not a composite pc. no extra info needs to be added

    var index=getMatIndex(recursionCounter);
    matInfo[index].string      += info;
    matInfo[index].stringshort += infoshort;

    //is a composite pc so there will be a div in the next position
    var generatedDiv="";
    generatedDiv = $("#"+pcListID).next().get(0).id; //this will be a div, eg. mg0_, bjacobi1_

    for (var i=0; i<$("#"+generatedDiv).children().length; i++) { //loop over all pcLists under this Div
	var childID = $("#"+generatedDiv).children().get(i).id;
	if ($("#"+childID).is(".pcLists")) {//has more pc lists that need to be taken care of recursively
            pcGetDetailedInfo(childID,prefix,recursionCounter,matInfo);
	}
    }
}




/*
  solverGetOptions - get the options from the drop-down lists
  input:
    matInfo
  output:
    matInfo[].string stores collected solver options
    matInfo[].stringshort
*/
function solverGetOptions(matInfo)
{
    var prefix,kspSelectedValue,pcSelectedValue,level;

    for (var i = 0; i<matInfo.length; i++) {
	if (typeof matInfo[i] != 'undefined' && matInfo[i].id != "-1") {
	    //get the ksp and pc options at the topest solver-level
	    kspSelectedValue = $("#kspList" + matInfo[i].id).val();
	    pcSelectedValue  = $("#pcList" + matInfo[i].id).val();

            //get prefix
            prefix = "-";
            // for pc=fieldsplit
            for (level=1; level<=matInfo[i].matLevel; level++) {
                if (level == matInfo[i].matLevel) {
                    if(matInfo[i].name == undefined)
                        prefix += "fieldsplit_A"+matInfo[i].id+"_"; // matInfo[i].id
                    else
                        prefix += "fieldsplit_"+matInfo[i].name+"_"; // use fsName if possible!!
                } else {
                    var parent = matInfo[i].id.substring(0,matInfo[i].id.length-1);//take everything except last char
                    var parentLevel = parent.length-1;//by definition. because A0 is length 1 but level 0
                    while (level < parentLevel) {
                        parent = parent.substring(0,parent.length-1);//take everything except last char
                        parentLevel = parent.length-1;
                    }
                    if(matInfo[getMatIndex(parent)].name == undefined)
                        prefix += "fieldsplit_A"+parent+"_";
                    else
                        prefix += "fieldsplit_"+matInfo[getMatIndex(parent)].name+"_";
                }
            }

	    //together, with the name, make a full string for printing
            matInfo[i].string = ("Matrix A" + matInfo[i].id + "<br /> "+prefix+"ksp_type " + kspSelectedValue + "<br />"+prefix+"pc_type " + pcSelectedValue);
            matInfo[i].stringshort = ("Matrix A" + matInfo[i].id + "<br /> KSP: " + kspSelectedValue + "; PC: " + pcSelectedValue);

            // for composite pc, get additional info from the rest of pcLists
            pcGetDetailedInfo("pcList"+ matInfo[i].id,prefix,matInfo[i].id,matInfo);
	}
    }
}