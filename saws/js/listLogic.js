$(document).on("keyup", '.processorInput', function() {
    if ($(this).val().match(/[^0-9]/) || $(this).val()==0) {//problem is that integer only bubble still displays when nothing is entered
	$(this).attr("title","hello");//set a random title (this will be overwritten)
	$(this).tooltip();//create a tooltip from jquery UI
	$(this).tooltip({ content: "Integer only!" });//edit displayed text
	$(this).tooltip("open");//manually open once
    } else {
	$(this).removeAttr("title");//remove title attribute
	$(this).tooltip("destroy");
    }
});

$(document).on("keyup", '.fieldsplitBlocks', function() {//alerts user with a tooltip when an invalid input is provided
    if ($(this).val().match(/[^0-9]/) || $(this).val()==0 || $(this).val()==1) {
	$(this).attr("title","hello");//set a random title (this will be overwritten)
	$(this).tooltip();//create a tooltip from jquery UI
	$(this).tooltip({content: "At least 2 blocks!"});//edit displayed text
	$(this).tooltip("open");//manually open once
    } else {
	$(this).removeAttr("title");//remove title attribute
	$(this).tooltip("destroy");
    }
});

/*
  This function is called when the drop-down menu of .pcLists is excuted
*/
$(document).on('change', '.pcLists', function(){

    //if(preRecursionCounter != -1)
        //alert("preRecursionCounter is not -1"); //preRecursionCounter is always -1 in the beginning when pulling data from SAWs

    //alert("beginning current:"+currentRecursionCounterSAWs);

    //get the pc option
    var pcValue = $(this).val();
    if (pcValue == null) alert("Warning: pcValue = null!");

    //.parent() returns a weird object so we need to use .get(0)
    var parentDiv = $(this).parent().get(0).id;

    //first, find parent (the number after the A)
    var parent = parentDiv;
    while (parent.indexOf('_') != -1)
	parent=$("#"+parent).parent().get(0).id;
    parent = parent.substring(1, parent.length);

    //alert('parentDiv '+ parentDiv + '; parent '+parent + '; pcValue '+pcValue +'; this.id '+ this.id+'; recursionCounterSAWs '+recursionCounterSAWs);

    if (parent == "-1") return; //endtag for o-1 and other oparent are not consistent yet???

    if (pcValue == "mg") {
        //------------------------------------------------------
	var newDiv = generateDivName(this.id,parent,"mg");
        var endtag = newDiv.substring(newDiv.indexOf('_'));
        var myendtag;
        var mgLevels;

	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:50px;'></div>");
        myendtag = endtag+"0";
	$("#"+newDiv).append("<b>MG Type &nbsp;&nbsp;</b><select class=\"mgList\" id=\"mgList" + parent +myendtag+"\"></select>");
        populateMgList("mgList"+parent+myendtag);

        // mglevels determines how many ksp/pc at this solve level  
        $("#"+newDiv).append("<br><b>MG Levels </b><input type='text' id=\'mglevels"+parent+myendtag+"\' maxlength='4' class='mgLevels'>");
        mgLevels = 2; //default
        if (preRecursionCounter == -1) {
            mgLevels=sawsInfo[currentRecursionCounterSAWs-1].mg_levels;//need to minus one because currentRecCounter starts at 1 while array starts at 0
        }
        $("#mglevels"+parent+myendtag).val(mgLevels); // set default mgLevels -- when reset mglevels to 4, gives order 3, 2, 1 below???

        // Coarse Grid Solver (Level 0)
        myendtag = endtag+"0";
	$("#"+newDiv).append("<br><br><b>Coarse Grid Solver (Level 0)  </b>");
        $("#"+newDiv).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"kspLists\" id=\"kspList" + parent+myendtag +"\"></select>");
	$("#"+newDiv).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList" + parent+myendtag +"\"></select>");

        var prefix = "";
        if (preRecursionCounter == -1) 
            prefix = sawsInfo[currentRecursionCounterSAWs].prefix;
        if (prefix == "mg_coarse_") { //use SAWs options;
            var SAWs_kspVal = $("#kspList-1"+prefix).val(); 
            var SAWs_pcVal = $("#pcList-1"+prefix).val(); 
            //alternative???
            populateKspList("kspList"+parent+myendtag,null,SAWs_kspVal);
            populatePcList("pcList"+parent+myendtag,null,SAWs_pcVal);
            currentRecursionCounterSAWs++;
	    $("#pcList"+parent+myendtag).trigger("change");
        } else {
	    populateKspList("kspList"+parent+myendtag,null,"null");
	    populatePcList("pcList"+parent+myendtag,null,"null");
	    //set defaults
	    $("#kspList"+parent+myendtag).find("option[value='preonly']").attr("selected","selected");
	    $("#pcList"+parent+myendtag).find("option[value='redundant']").attr("selected","selected");
	    //redundant has to have extra dropdown menus so manually trigger
	    $("#pcList"+parent+myendtag).trigger("change");
        }

        // Smoothing (Level>0)
        $("#"+newDiv).append("<br><br><b id=\"text_smoothing"+parent+endtag+"\">Smoothing   </b>")
        mgLevels = $("#mglevels" + parent + myendtag).val();

        if (mgLevels > 1) {
            for (var level=1; level<mgLevels; level++) { 
                myendtag = endtag+level;
                $("#"+newDiv).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b id=\"text_kspList"+parent+myendtag+"\">KSP Level "+level+" &nbsp;&nbsp;</b><select class=\"kspLists\" id=\"kspList"+ parent+myendtag +"\"></select>");
	        $("#"+newDiv).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b id=\"text_pcList"+parent+myendtag+"\">PC Level "+level+" &nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList"+ parent+myendtag+"\"></select>");
                prefix = "";
                if (preRecursionCounter == -1)
                    prefix = sawsInfo[currentRecursionCounterSAWs].prefix;
                if (prefix == "mg_levels_"+level+"_") { //use SAWs options
                    //alert("prefix "+prefix+" match");
                    var SAWs_kspVal = $("#kspList-1"+prefix).val();
                    var SAWs_pcVal = $("#pcList-1"+prefix).val();
                    //alternative???
                    populateKspList("kspList"+parent+myendtag,null,SAWs_kspVal);
                    populatePcList("pcList"+parent+myendtag,null,SAWs_pcVal);
                    currentRecursionCounterSAWs++;
	            $("#pcList"+parent+myendtag).trigger("change");
                } else {
                    populateKspList("kspList"+parent+myendtag,null,"null");
	            populatePcList("pcList"+parent+myendtag,null,"null"); 
                    // set defaults
                    $("#kspList"+parent+myendtag).find("option[value='chebyshev']").attr("selected","selected");
	            $("#pcList"+parent+myendtag).find("option[value='sor']").attr("selected","selected");
                }
            }
        } 

    } else { //if not mg, remove the options that mg might have added
	var newDiv=generateDivName(this.id,parent,"mg");
	$("#"+newDiv).remove();
    }

    if (pcValue == "redundant") {
        //------------------------------------------------------
	var newDiv=generateDivName(this.id,parent,"redundant");
	var endtag=newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
	
	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:50px;'></div>");
	//text input box
        var myendtag = endtag+"0"; // enable different ksp/pc for each redundant number
	$("#"+newDiv).append("<b>Redundant number   </b><input type='text' id='redundantNumber"+parent+myendtag+"\' value='np' maxlength='4' class='processorInput'>");
	$("#"+newDiv).append("<br><b>Redundant KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"redundant\" id=\"kspList" + parent +myendtag+"\"></select>");
	$("#"+newDiv).append("<br><b>Redundant PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList" + parent +myendtag+"\"></select>");
	populateKspList("kspList"+parent+myendtag,null,"null");
	populatePcList("pcList"+parent+myendtag,null,"null");

	//set defaults for redundant
	$("#kspList"+parent+myendtag).find("option[value='preonly']").attr("selected","selected");
        var index=getMatIndex(parent);
        if (matInfo[index].symm) {
            $("#pcList"+parent+myendtag).find("option[value='cholesky']").attr("selected","selected");
        } else {
	    $("#pcList"+parent+myendtag).find("option[value='lu']").attr("selected","selected");
        }
    } else { //remove dropdown lists associated with redundant
	var newDiv=generateDivName(this.id,parent,"redundant");
	$("#"+newDiv).remove();
    }

    if (pcValue == "bjacobi") {
        //------------------------------------------------------
	var newDiv = generateDivName(this.id,parent,"bjacobi");
	var endtag = newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
        //alert("bjacobi: newDiv="+newDiv);
	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:50px;'></div>");
      
	//text input box
        var myendtag = endtag+"0"; // enable different ksp/pc for each redundant number
	$("#"+newDiv).append("<b>Bjacobi blocks </b><input type='text' id='bjacobiBlocks"+parent+myendtag+"\' value='np' maxlength='4' class='processorInput'>"); // use style='margin-left:30px;'
	$("#"+newDiv).append("<br><b>Bjacobi KSP   &nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"kspLists\" id=\"kspList"+parent+myendtag+"\"></select>");
	$("#"+newDiv).append("<br><b>Bjacobi PC   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList"+parent+myendtag+"\"></select>");
        if (preRecursionCounter == -1) { //use SAWs options
            var prefix = sawsInfo[currentRecursionCounterSAWs].prefix;//not minus one because we need to skip one (already did the first ksp/pc in main.js)
            var SAWs_kspVal = $("#kspList-1"+prefix).val();
            var SAWs_pcVal = $("#pcList-1"+prefix).val();
            //alert("using prefix:"+prefix+", using ksp:"+SAWs_kspVal+", using pc:"+SAWs_pcVal);
            //alternative???
            populateKspList("kspList"+parent+myendtag,null,SAWs_kspVal);
            populatePcList("pcList"+parent+myendtag,null,SAWs_pcVal);
            if(typeof sawsInfo[currentRecursionCounterSAWs-1].bjacobi_blocks == "undefined")//sometimes SAWs will fail to come up with a default
                sawsInfo[currentRecursionCounterSAWs-1].bjacobi_blocks = "saws undefined";//this makes sure the program doesn't crash
            $("#bjacobiBlocks"+parent+myendtag).attr("value",sawsInfo[currentRecursionCounterSAWs-1].bjacobi_blocks);//minus 1 because array starts from 0 while currentRecursionCounter starts from 1
            //alert("bjacobi: prefix="+prefix+"; SAWs_kspVal="+SAWs_kspVal+"; SAWs_pcVal="+SAWs_pcVal+"; currentRecursionCounterSAWs="+currentRecursionCounterSAWs);
            currentRecursionCounterSAWs++;
            //manually trigger pclist once
	    $("#pcList"+parent+myendtag).trigger("change");
        } else {
	    populateKspList("kspList"+parent+myendtag,null,"null");
            populatePcList("pcList"+parent+myendtag,null,"null");

	    //set defaults for bjacobi
	    $("#kspList"+parent+myendtag).find("option[value='preonly']").attr("selected","selected");
            var index=getMatIndex(parent);
            if (matInfo[index].symm) {
                $("#pcList"+parent+myendtag).find("option[value='icc']").attr("selected","selected");
            } else {
	        $("#pcList"+parent+myendtag).find("option[value='ilu']").attr("selected","selected");
            }
        }

        // if parentDiv = bjacobi, it is a Hierarchical Krylov method, display an image for illustration
        var parentDiv_str = parentDiv.substring(0,7);
        if (parentDiv_str == pcValue) {
            alert("parentDiv_str "+ parentDiv_str +" = pcValue, Hierarchical Krylov Method - display an image of the sovler!");
        }
    }
    else {//not bjacobi - why generate, then remove???(this is because if someone JUST switched from bjacobi to something else, then bjacobi would be removed)
	var newDiv = generateDivName(this.id,parent,"bjacobi");
	$("#"+newDiv).remove();
    }

    if (pcValue == "asm") {
        //------------------------------------------------------
	var newDiv=generateDivName(this.id,parent,"asm");
	var endtag=newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:50px;'></div>");

	//text input box
        var myendtag = endtag+"0"; // enable different ksp/pc for each redundant number
        $("#"+newDiv).append("<b>ASM blocks   &nbsp;&nbsp;</b><input type='text' id='asmBlocks"+parent+myendtag+"\' value='np' maxlength='4'>");
	$("#"+newDiv).append("<br><b>ASM overlap   </b><input type='text' id='asmOverlap"+parent+myendtag+"\' value='1' maxlength='4'>");
	$("#"+newDiv).append("<br><b>ASM KSP   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"kspLists\" id=\"kspList" + parent +myendtag+"\"></select>");
	$("#"+newDiv).append("<br><b>ASM PC   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList" + parent +myendtag+"\"></select>");
	populateKspList("kspList"+parent+myendtag,null,"null");
	populatePcList("pcList"+parent+myendtag,null,"null");

	//set defaults for asm
	$("#kspList"+parent+myendtag).find("option[value='preonly']").attr("selected","selected");
        var index=getMatIndex(parent);
        if (matInfo[index].symm) {
            $("#pcList"+parent+myendtag).find("option[value='icc']").attr("selected","selected");
        } else {
	    $("#pcList"+parent+myendtag).find("option[value='ilu']").attr("selected","selected");
        }
    } else {//not asm
	var newDiv = generateDivName(this.id,parent,"asm");
	$("#"+newDiv).remove();
    }

    if (pcValue == "ksp") {
        //------------------------------------------------------
	var newDiv = generateDivName(this.id,parent,"ksp");
	var endtag = newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:50px;'></div>");

	//text input box
        var myendtag = endtag+"0";
	$("#"+newDiv).append("<b>KSP KSP   </b><select class=\"kspLists\" id=\"kspList" + parent +myendtag+"\"></select>");
	$("#"+newDiv).append("<br><b>KSP PC &nbsp;&nbsp; </b><select class=\"pcLists\" id=\"pcList" + parent +myendtag+"\"></select>");

        if (preRecursionCounter == -1) { //use SAWs options
            var prefix = sawsInfo[currentRecursionCounterSAWs].prefix;
            var SAWs_kspVal = $("#kspList-1"+prefix).val();
            var SAWs_pcVal = $("#pcList-1"+prefix).val();
            //alternative???
            populateKspList("kspList"+parent+myendtag,null,SAWs_kspVal);
            populatePcList("pcList"+parent+myendtag,null,SAWs_pcVal);
            currentRecursionCounterSAWs++;
            $("#pcList"+parent+myendtag).trigger("change");
        } else {
	    populateKspList("kspList"+parent+myendtag,null,"null");
	    populatePcList("pcList"+parent+myendtag,null,"null");

	    //set defaults for ksp
	    $("#kspList"+parent+myendtag).find("option[value='gmres']").attr("selected","selected");
	    $("#pcList"+parent+myendtag).find("option[value='bjacobi']").attr("selected","selected");
	    //bjacobi has extra dropdown menus so manually trigger once
	    $("#pcList"+parent+myendtag).trigger("change");
        }

        // if parentDiv = ksp, it is a Nested Krylov method, display an image for illustration
        var parentDiv_str = parentDiv.substring(0,3);
        if (parentDiv_str == pcValue) {
            alert('parentDiv_str '+ parentDiv_str +' = pcValue, Neste Krylov Method - display an image of the sovler!');
        }
    } else {//not ksp
	var newDiv = generateDivName(this.id,parent,"ksp");
	$("#"+newDiv).remove();
    }

    if (pcValue == "fieldsplit") {
        //------------------------------------------------------
        var index=getMatIndex(parent);
        if (!matInfo[index].logstruc) {
            alert("Error: Preconditioner fieldsplit cannot be used for non-logically blocked matrix!");//how to throw an error???
            return;//no new divs will be generated
        }

        var newDiv = generateDivName(this.id,parent,"fieldsplit");//this div contains the two fieldsplit dropdown menus
	var endtag = newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:"+50+"px;'></div>");
        myendtag = endtag+"0";
	$("#"+newDiv).append("<b>Fieldsplit Type &nbsp;&nbsp;</b><select class=\"fieldsplitList\" id=\"fieldsplitList" + parent +myendtag+"\"></select>");
        $("#"+newDiv).append("<br><b>Fieldsplit Blocks </b><input type='text' id='fieldsplitBlocks"+parent+myendtag+"\' value='2' maxlength='2' class='fieldsplitBlocks'>");//notice that the class is now fieldsplitBlocks instead of fieldsplitBlocksInput
        populateFieldsplitList("fieldsplitList"+parent+myendtag,null,"null");

        for(var i=1; i>=0; i--) {//is indeed logstruc so by default append two A divs
            var index=getMatIndex(parent);//properties (symm, posdef, etc, are inherited from parent). logstruc property is set to false.
            var newChild=parent+""+i;
            var indentation = 30 * (newChild.length-1);//minus one because length 1 is level 0
            $("#row"+parent).after("<tr id='row"+newChild+"'> <td> <div style=\"margin-left:"+indentation+"px;\" id=\"A"+ newChild + "\" title=\"A"+ newChild + " Symm:"+matInfo[index].symm+" Posdef:"+matInfo[index].posdef+" Logstruc:"+false+"\"> </div></td> <td> <div id=\"oCmdOptions" + parent + "\"></div> </td> </tr>");

            //todo: write info for the parent A div

            matInfo[matInfoWriteCounter] = {
            posdef:  matInfo[index].posdef,
            symm:    matInfo[index].symm,
            logstruc:false,
            blocks: 2,
            matLevel:   newChild.length-1,
            id:       newChild
	}

            matInfoWriteCounter++;

            $("#A" + newChild).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>KSP &nbsp;</b><select class=\"kspLists\" id=\"kspList" + newChild +"\"></select>");
	    $("#A" + newChild).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>PC &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList" + newChild +"\"></select>");
            populateKspList("kspList"+newChild,null,"null");
            populatePcList("pcList"+newChild,null,"null");
            $("#pcList"+newChild).trigger("change");//make sure necessary options are added on
        }

        //the function that reacts to change in fieldsplit blocks adds and removes A divs as necessary
    }
    else { //not fieldsplit

        //todo: delete info (if any) for this A div

	var newDiv = generateDivName(this.id,parent,"fieldsplit");
	$("#"+newDiv).remove();

        // for the whole thing, remove A divs as necessary since not fieldsplit
        var index=getMatIndex(parent);
        if($(this).attr("id").indexOf('_')==-1 && matInfo[index].logstruc) {//if top level pclist AND the A matrix is logically structured...needs removal
            removeChildren(parent);//recursive function that removes all children of a particular A div
        }
    }

});

//input: id of A matrix
//output: divs of children of that A matrix are removed. places in matInfo where they are stored are wasted (e.g. -1 is put into the 'id' so that slot can never be used again)
function removeChildren(id) {

    var index=getMatIndex(id);
    var numChildren=matInfo[index].blocks;
    for(var i=0; i<numChildren; i++) {
        var child=""+id+i;
        index=getMatIndex(child);
        if(matInfo[index].fieldsplitBlocks>0)//this child has more children
        {
            removeChildren(child);//recursive call to remove all children of that child
        }
        matInfo[index].id="-1";//make sure this location is never accessed again.
        $("#A"+child).remove();//remove that child itself
        $("#row"+child).remove();//remove the row in the oContainer table
    }
}

/*
  generateDivName - generate a div name
  input:
    id - the id of the current dropdown menu
    matRecursion - matrix recursion counter (the number after the o)
    pcValue - e.g., bjacobi, mg, redundant
  output:
    newDiv - name of new Div in the format "pcValue + matRecursion + endtag + ...", 
             eg. bjacobi0_0, ksp0_00, mg0_
*/
function generateDivName(id,matRecursion,pcValue) 
{
    var newDiv;
    var endtag = id.substring(id.lastIndexOf("_"), id.length);
    //alert("generateDivName, pcValue "+pcValue+"; id "+id+"; endtag "+endtag);

    if (id.indexOf("_") == -1) { //A div
        newDiv = pcValue + matRecursion + "_";
    } else {
        newDiv = pcValue + matRecursion + endtag;
    } 
    //alert("newDiv "+newDiv);
    return newDiv;
}

//called when text input "fieldsplitBlocks" is changed
$(document).on('change', '.fieldSplitBlocksInput', function() {

//todo: this needs a lot of work

});

/*
  This function is called when the text input "MG Levels" is changed  
*/
$(document).on('change', '.mgLevels', function()
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
        for (var level=1; level<mgLevels; level++) { // must input text_smoothing in reverse order, i.e., PC, KSP, ..., PC, KSP - don't know why???

	    if (level<10)//still using numbers
		myendtag = endtag+level;
	    else
		myendtag = endtag+'abcdefghijklmnopqrstuvwxyz'.charAt(level-10);//add the correct char

	    $("#text_smoothing"+recursionCounter+endtag).after("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b id=\"text_pcList"+recursionCounter+myendtag+"\">PC Level "+level+" &nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList"+ recursionCounter+myendtag+"\"></select>");
            $("#text_smoothing"+recursionCounter+endtag).after("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b id=\"text_kspList"+recursionCounter+myendtag+"\">KSP Level "+level+" &nbsp;&nbsp;</b><select class=\"kspLists\" id=\"kspList"+ recursionCounter+myendtag +"\"></select>");

            populateKspList("kspList"+recursionCounter+myendtag,null,"null");
	    populatePcList("pcList"+recursionCounter+myendtag,null,"null");
            // set defaults
            $("#kspList"+recursionCounter+myendtag).find("option[value='chebyshev']").attr("selected","selected");
	    $("#pcList"+recursionCounter+myendtag).find("option[value='sor']").attr("selected","selected");
        }
    }
});

//copied from main.js
//input: desired id in string format. (for example, "01001")
//output: index in matInfo where information on that id is located
function getMatIndex(id)
{
    //matInfo and matInfoWriteCounter are visible here although declared in main.js
    for(var i=0; i<matInfoWriteCounter; i++) {
        if(matInfo[i].id == id)
            return i;//return index where information is located.
    }

    return -1;//invalid id.
}