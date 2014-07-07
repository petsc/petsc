/*
  This function is called when a pc_type option is changed (new options may need to be displayed and/or old ones removed
*/

$(document).on("change","select[id^='pc_type']",function() {

    //get the pc option
    var pcValue   = $(this).val();
    var id        = $(this).attr("id");//really should not be used in this method. there are better ways of getting information
    var endtag    = id.substring(id.indexOf("0"),id.length);
    var parentDiv = "solver" + endtag;

    removeAllChildren(endtag);//this function also changes matInfo as needed

    if (pcValue == "mg") {
        var defaultMgLevels = 2;

        //first add options related to multigrid (pc_mg_type and pc_mg_levels)
        $("#" + parentDiv).append("<br><b>MG Type &nbsp;&nbsp;</b><select id=\"pc_mg_type" + endtag + "\"></select>");
        $("#" + parentDiv).append("<br><b>MG Levels </b><input type='text' id=\'pc_mg_levels" + endtag + "\' maxlength='4'>");
        $("#pc_mg_levels" + endtag).val(defaultMgLevels);

        populateMgList(endtag);

        //also record new information in matInfo
        for(var i=defaultMgLevels-1; i>=0; i--) {
            var childEndtag = endtag + "_" + i;

            var writeLoc = matInfo.length;
            matInfo[writeLoc] = {
                pc_type : "",
                ksp_type: "",
                endtag : childEndtag
            }

            var margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

            $("#" + parentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
            if(i == 0) //coarse grid solver (level 0)
                $("#solver" + childEndtag).append("<br><b>Coarse Grid Solver (Level 0)  </b>");
            else
                $("#solver" + childEndtag).append("<br><b>Smoothing   </b>");

            $("#solver" + childEndtag).append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag  + "\"></select>");
	    $("#solver" + childEndtag).append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

            populateKspList(childEndtag);
            populatePcList(childEndtag);

	    //set defaults
	    $("#ksp_type" + childEndtag).find("option[value='preonly']").attr("selected","selected");
	    $("#pc_type" + childEndtag).find("option[value='redundant']").attr("selected","selected");
	    //redundant has to have extra dropdown menus so manually trigger
	    $("#pc_type" + childEndtag).trigger("change");
        }

    }

    else if (pcValue == "redundant") {
        var defaultRedundantNumber = 2;
        var childEndtag = endtag + "_0";

        //first add options related to redundant (pc_redundant_number)
        $("#" + parentDiv).append("<br><b>Redundant Number </b><input type='text' id=\'pc_redundant_number" + endtag + "\' maxlength='4'>");
        $("#pc_redundant_number" + endtag).val(defaultRedundantNumber);

        var writeLoc = matInfo.length;
        matInfo[writeLoc] = {
            pc_type : "",
            ksp_type: "",
            endtag : childEndtag,
            symm: false
        }

        var margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

	$("#" + parentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");

	$("#solver" + childEndtag).append("<br><b>Redundant Solver Options </b>");
	$("#solver" + childEndtag).append("<br><b>KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag + "\"></select>");
	$("#solver" + childEndtag).append("<br><b>PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

	populateKspList(childEndtag);
        populatePcList(childEndtag);

	//set defaults for redundant
	$("#ksp_type" + childEndtag).find("option[value='preonly']").attr("selected","selected");
        var index = getIndex(matInfo,endtag);
        if (matInfo[index].symm) {
            $("#pc_type" + childEndtag).find("option[value='cholesky']").attr("selected","selected");
        } else {
	    $("#pc_type" + childEndtag).find("option[value='lu']").attr("selected","selected");
        }
    }

    else if (pcValue == "bjacobi") {

        var defaultBjacobiBlocks = 2;
        var childEndtag = endtag + "_0";

        //first add options related to bjacobi (pc_bjacobi_blocks)
        $("#" + parentDiv).append("<br><b>Bjacobi Blocks </b><input type='text' id=\'pc_bjacobi_blocks" + endtag + "\' maxlength='4'>");
        $("#pc_bjacobi_blocks" + endtag).val(defaultBjacobiBlocks);

        var writeLoc = matInfo.length;
        matInfo[writeLoc] = {
            pc_type : "",
            ksp_type: "",
            endtag : childEndtag,
            symm: false
        }

        var margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

	$("#" + parentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");

	$("#solver" + childEndtag).append("<br><b>Bjacobi Solver Options </b>");
	$("#solver" + childEndtag).append("<br><b>KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag + "\"></select>");
	$("#solver" + childEndtag).append("<br><b>PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

	populateKspList(childEndtag);
        populatePcList(childEndtag);

	//set defaults for bjacobi
	$("#ksp_type" + childEndtag).find("option[value='preonly']").attr("selected","selected");
        var index=getIndex(matInfo, endtag);
        if (matInfo[index].symm) {
            $("#pc_type" + childEndtag).find("option[value='icc']").attr("selected","selected");
        } else {
	    $("#pc_type" + childEndtag).find("option[value='ilu']").attr("selected","selected");
        }
    }

    else if (pcValue == "asm") {

        var defaultAsmBlocks  = 2;
        var defaultAsmOverlap = 2;
        var childEndtag = endtag + "_0";

        //first add options related to ASM
        $("#" + parentDiv).append("<br><b>ASM blocks   &nbsp;&nbsp;</b><input type='text' id=\"pc_asm_blocks" + endtag + "\" maxlength='4'>");
	$("#" + parentDiv).append("<br><b>ASM overlap   </b><input type='text' id=\"pc_asm_overlap" + endtag + "\" maxlength='4'>");
        $("#pc_asm_blocks" + endtag).val(defaultAsmBlocks);
        $("#pc_asm_overlap" + endtag).val(defaultAsmOverlap);

        var writeLoc = matInfo.length;
        matInfo[writeLoc] = {
            pc_type : "",
            ksp_type: "",
            endtag : childEndtag,
            symm: false
        }

        var margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

	$("#" + parentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");

	$("#solver" + childEndtag).append("<br><b>ASM Solver Options </b>");
	$("#solver" + childEndtag).append("<br><b>KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag + "\"></select>");
	$("#solver" + childEndtag).append("<br><b>PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

	populateKspList(childEndtag);
        populatePcList(childEndtag);

	//set defaults for asm
	$("#ksp_type" + childEndtag).find("option[value='preonly']").attr("selected","selected");
        var index = getIndex(matInfo, endtag);
        if (matInfo[index].symm) {
            $("#pc_type" + childEndtag).find("option[value='icc']").attr("selected","selected");
        } else {
	    $("#pc_type" + childEndtag).find("option[value='ilu']").attr("selected","selected");
        }
    }

    else if (pcValue == "ksp") {
        var childEndtag = endtag + "_0";

        var writeLoc = matInfo.length;
        matInfo[writeLoc] = {
            pc_type : "",
            ksp_type: "",
            endtag : childEndtag,
            symm: false
        }

        var margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

	$("#" + parentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");

	$("#solver" + childEndtag).append("<br><b>KSP Solver Options </b>");
	$("#solver" + childEndtag).append("<br><b>KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag + "\"></select>");
	$("#solver" + childEndtag).append("<br><b>PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

	populateKspList(childEndtag);
        populatePcList(childEndtag);

	//set defaults for ksp
	$("#ksp_type" + childEndtag).find("option[value='gmres']").attr("selected","selected");
	$("#pc_type" + childEndtag).find("option[value='bjacobi']").attr("selected","selected");
	//bjacobi has extra dropdown menus so manually trigger once
	$("#pc_type" + childEndtag).trigger("change");
    }

    else if (pcValue == "fieldsplit") {//just changed to fieldsplit so set up defaults (2 children)

        var defaultFieldsplitBlocks = 2;

        //first add options related to fieldsplit (pc_fieldsplit_type and pc_fieldsplit_blocks)
        $("#" + parentDiv).append("<br><b>Fieldsplit Type &nbsp;&nbsp;</b><select id=\"pc_fieldsplit_type" + endtag + "\"></select>");
        $("#" + parentDiv).append("<br><b>Fieldsplit Blocks </b><input type='text' id=\"pc_fieldsplit_blocks" + endtag + "\" maxlength='4'>");
        $("#pc_fieldsplit_blocks" + endtag).val(defaultFieldsplitBlocks);

        populateFieldsplitList(endtag);

        //also record new information in matInfo
        for(var i=defaultFieldsplitBlocks-1; i>=0; i--) {
            var childEndtag = endtag + "_" + i;

            var writeLoc = matInfo.length;
            matInfo[writeLoc] = {
                pc_type : "",
                ksp_type: "",
                endtag : childEndtag
            }

            var margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

            $("#" + parentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
            $("#solver" + childEndtag).append("<br><b>Fieldsplit " + (i+1) + " Options</b>");
            $("#solver" + childEndtag).append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag  + "\"></select>");
	    $("#solver" + childEndtag).append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

            populateKspList(childEndtag);
            populatePcList(childEndtag);

	    //set defaults
	    $("#ksp_type" + childEndtag).find("option[value='preonly']").attr("selected","selected");
	    $("#pc_type" + childEndtag).find("option[value='redundant']").attr("selected","selected");
	    //redundant has to have extra dropdown menus so manually trigger
	    $("#pc_type" + childEndtag).trigger("change");
        }
    }
});

//input: endtag of the parent
function removeAllChildren(endtag) {

    var index       = getIndex(matInfo, endtag);
    var numChildren = getNumChildren(matInfo, endtag);

    for(var i=0; i<numChildren; i++) {
        var childEndtag = endtag + "_" + i;
        var childIndex  = getIndex(matInfo,childEndtag);

        if(getNumChildren(matInfo, childEndtag) > 0)//this child has more children
        {
            removeAllChildren(childEndtag);//recursive call to remove all children of that child
        }
        matInfo[childIndex].endtag = "-1";//make sure this location is never accessed again.

        $("#solver" + childEndtag).remove();//remove that child itself
    }

    //adjust variables in matInfo
    if(matInfo[index].pc_type == "mg") {
        matInfo[index].pc_mg_levels = 0;
    }
    else if(matInfo[index].pc_type == "fieldsplit") {
        matInfo[index].pc_fieldsplit_blocks = 0;
    }

    $("#pc_type" + endtag).nextAll().remove();//remove the options in the same level solver

}

//called when text input for pc_fieldsplit_blocks is changed
$(document).on('keyup', "select[id^='pc__fieldsplit_blocks']", function() {

    if($(this).val().match(/[^0-9]/) || $(this).val()<1) //return on invalid input
        return;

    var id     = this.id;
    var endtag = id.substring(id.indexOf(0),id.length);
    var index  = getIndex(matInfo, endtag);
    var val    = $(this).val();

    // this next part is a bit tricky

    if( ) //case 1: we need to remove some divs



    //case 1: more A divs need to be added
    if(val > matInfo[index].blocks) {
        for(var i=val-1; i>=matInfo[index].blocks; i--) {//insert in backwards order for convenience
            //add divs and write matInfo

            var indentation=(parent.length-1+1)*30; //according to the length of currentAsk (aka matrix level), add margins of 30 pixels accordingly
            $("#row"+lastBlock).after("<tr id='row"+parent+i+"'> <td> <div style=\"margin-left:"+indentation+"px;\" id=\"A"+ parent+i + "\"> </div></td> <td> <div id=\"oCmdOptions" + parent+i + "\"></div> </td> </tr>");

            //Create drop-down lists. '&nbsp;' indicates a space
            var newChild = parent + i;

            $("#A" + parent+i).append("<br><b id='matrixText"+parent+i+"'>A" + "<sub>" + parent+i + "</sub>" + " (Symm:"+matInfo[index].symm+" Posdef:"+matInfo[index].posdef+" Logstruc:false)</b>");
	    $("#A" + parent+i).append("<br><b>KSP &nbsp;</b><select class=\"kspLists\" id=\"kspList" + parent+i +"\"></select>");
	    $("#A" + parent+i).append("<br><b>PC &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList" + parent+i +"\"></select>");


            //populate the kspList and pclist with default options
            populateKspList("kspList"+parent+i,null,"null");
            populatePcList("pcList"+parent+i,null,"null");

            $("#pcList"+parent+i).trigger("change");

            var writeLoc = matInfo.length;
            matInfo[writeLoc] = {
                posdef:  matInfo[index].posdef,//inherits attributes of parents
                symm:    matInfo[index].symm,
                logstruc:false,//HOW SHOULD THIS BE ADDRESSED ?????????
                blocks: 0,//children do not have further children
                matLevel:   parent.length-1+1,
                id:       parent+i
            }

        }
        matInfo[index].blocks=val;
    }

    //case 2: some A divs need to be removed
    else if(val < matInfo[index].blocks) {
        for(var i=val; i<matInfo[index].blocks; i++) {
            removeChildren(parent+""+i);//remove grandchildren if any exist
            matInfo[getMatIndex(parent+""+i)].id="-1";//set matInfo id to -1
            $("#A"+parent+""+i).remove(); //manually remove the extras themselves
            $("#row"+parent+""+i).remove();
        }
        matInfo[index].blocks=val;
    }

});

/*
  This function is called when the text input "MG Levels" is changed
*/
$(document).on('keyup', '.mgLevels', function()
{
    //get mgLevels
    var mgLevels = $(this).val();
    if (mgLevels < 1) alert("Error: mgLevels must be >= 1!");

    // get parent div's id
    var newDiv           = $(this).parent().get(0).id; //eg., mg0_
    var loc              = newDiv.indexOf('_');

    //new way of finding parent (the id of the A matrix)
    var parentDiv = $(this).parent().get(0).id;
    while (parentDiv.indexOf('_') != -1)
	parentDiv=$("#"+parentDiv).parent().get(0).id;
    var recursionCounter = parentDiv.substring(1, parentDiv.length); //will work when there is more than 1 digit after 'A'

    var endtag = newDiv.substring(loc);
    //alert("newDiv "+newDiv+"; endtag "+endtag+"; recursionCounter "+recursionCounter);

    //instead of removing entire div, only remove the necessary smoothing options
    var ksp = $('b[id^="text_kspList'+recursionCounter+endtag+'"]').filter(function() {
	return this.id.substring(this.id.lastIndexOf('_'),this.id.length).length > endtag.length; //used to prevent removing options from higher levels since the first few characters would indeed match
    });

    ksp.next().next().remove();//remove br
    ksp.next().remove();//remove dropdown menus
    ksp.remove();//remove text itself

    var pc = $('b[id^="text_pcList'+recursionCounter+endtag+'"]').filter(function() {
	return this.id.substring(this.id.lastIndexOf('_'),this.id.length).length > endtag.length;
    });

    pc.next().next().remove();//remove br
    pc.next().remove();//remove dropdown menus
    pc.remove();//remove text itself

    var myendtag;
    //alert("mg: #pcList"+recursionCounter+endtag);
    if (endtag == "_") { // this is ugly! rename solver-level 0 kspList0 and pcList0 as kspList0_ and pcList0_ ???
        myendtag = "";
    } else {
        myendtag= endtag;
    }
    myendtag = endtag+"0";

    // Smoothing (Level>0)
    mgLevels = $("#mglevels" + recursionCounter + myendtag).val();
    if (mgLevels > 1) {
        for (var level=mgLevels-1; level>=1; level--) {

	    if (level<10)//still using numbers
		myendtag = endtag+level;
	    else
		myendtag = endtag+'abcdefghijklmnopqrstuvwxyz'.charAt(level-10);//add the correct char

	    $("#text_smoothing"+recursionCounter+endtag).after("<br><b id=\"text_pcList"+recursionCounter+myendtag+"\">PC Level "+level+" &nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList"+ recursionCounter+myendtag+"\"></select>");
            $("#text_smoothing"+recursionCounter+endtag).after("<br><b id=\"text_kspList"+recursionCounter+myendtag+"\">KSP Level "+level+" &nbsp;&nbsp;</b><select class=\"kspLists\" id=\"kspList"+ recursionCounter+myendtag +"\"></select>");

            var endtagEdit=myendtag.substring(1,myendtag.length);//take off the first character (the underscore)

            populateKspList("kspList"+recursionCounter+myendtag,null,"null");
	    populatePcList("pcList"+recursionCounter+myendtag,null,"null");
            // set defaults
            $("#kspList"+recursionCounter+myendtag).find("option[value='chebyshev']").attr("selected","selected");
	    $("#pcList"+recursionCounter+myendtag).find("option[value='sor']").attr("selected","selected");
        }
    }
});