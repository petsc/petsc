$(document).on("keyup", '.processorInput', function() {
    if ($(this).val().match(/[^0-9]/)) {
	$(this).attr("title","hello");//set a random title (this will be overwritten)
	$(this).tooltip();//create a tooltip from jquery UI
	$(this).tooltip({ content: "Integer only!" });//edit displayed text
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

    //get the pc option
    var pcValue = $(this).val();
    if (pcValue == null) alert("Warning: pcValue = null!");

    //.parent() returns a weird object so we need to use .get(0)
    var parentDiv = $(this).parent().get(0).id;
    
    //new way of finding parent (the number after o in the o div): parent=listRecursionCounter, but listRecursionCounter may not be defined, so cannot be used :-(
    parent = parentDiv;
    while (parent.indexOf('_') != -1)
	parent=$("#"+parent).parent().get(0).id;
    parent = parent.substring(1, parent.length);  //parent=matrix recursion counter b/c resursion counter is not in this scope
    //alert('parentDiv '+ parentDiv + '; parent '+parent + '; pcValue '+pcValue +'; this.id '+ this.id);
    //alert('logstruc='+matrixInformation[parent].logstruc);

    // if pcValue is changed to !fieldsplit for logically structured matrix
    if (pcValue != "fieldsplit" && matrixInformation[parent].logstruc) {
        // find indices of all its children, remove all the options of its children
	var children = [];
        var numChildren = matGetChildren(parent, maxMatricies, children);

	// set parentFieldSplit as false for its children
        for (var i=0; i< numChildren; i++) {
            if ($("#pcList" + children[i]).data("parentFieldSplit")) {
                $("#pcList" + children[i]).data("parentFieldSplit",false);
            }
        }
    }

    if (pcValue == "mg") {
        //------------------------------------------------------
	var newDiv = generateDivName(this.id,parent,"mg");
        var endtag = newDiv.substring(newDiv.indexOf('_'));
        var myendtag;
        //alert("mg: newDiv "+newDiv+"; endtag "+endtag);

	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:50px;'></div>");
        myendtag = endtag+"0";
	$("#"+newDiv).append("<b>MG Type &nbsp;&nbsp;</b><select class=\"mgList\" id=\"mgList" + parent +myendtag+"\"></select>")  
        populateMgList("mgList"+parent+myendtag);

        // mglevels determines how many ksp/pc at this solve level
        $("#"+newDiv).append("<br><b>MG Levels </b><input type='text' id=\'mglevels"+parent+myendtag+"\' maxlength='4' class='mgLevels'>");
        $("#mglevels"+parent+myendtag).val(2); // set default mgLevels
        //$("#mglevels"+parent+endtag).trigger("change");

        // Smoothing (Level>0)
        $("#"+newDiv).append("<br><br><b id=\"text_smoothing"+parent+endtag+"\">Smoothing   </b>")
        var mgLevels = $("#mglevels" + parent + myendtag).val();
       
        if (mgLevels > 1) {
            for (var level=mgLevels-1; level>0; level--) {
                myendtag = endtag+level;
                $("#"+newDiv).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b id=\"text_kspList"+parent+myendtag+"\">KSP Level "+level+" &nbsp;&nbsp;</b><select class=\"kspLists\" id=\"kspList"+ parent+myendtag +"\"></select>")
	        $("#"+newDiv).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b id=\"text_pcList"+parent+myendtag+"\">PC Level "+level+" &nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList"+ parent+myendtag+"\"></select>")
                populateKspList("kspList"+parent+myendtag,null,"null");
	        populatePcList("pcList"+parent+myendtag,null,"null"); 
                // set defaults
                $("#kspList"+parent+myendtag).find("option[value='chebyshev']").attr("selected","selected");
	        $("#pcList"+parent+myendtag).find("option[value='sor']").attr("selected","selected");
            }
        } 

        // Coarse Grid Solver (Level 0)
        myendtag = endtag+"0";
	$("#"+newDiv).append("<br><br><b>Coarse Grid Solver (Level 0)  </b>")
        $("#"+newDiv).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"kspLists\" id=\"kspList" + parent+myendtag +"\"></select>")
	$("#"+newDiv).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select class=\"pcLists\" id=\"pcList" + parent+myendtag +"\"></select>")
	populateKspList("kspList"+parent+myendtag,null,"null");
	populatePcList("pcList"+parent+myendtag,null,"null");
	//set defaults
	$("#kspList"+parent+myendtag).find("option[value='preonly']").attr("selected","selected");
	$("#pcList"+parent+myendtag).find("option[value='redundant']").attr("selected","selected");
	//redundant has to have extra dropdown menus so manually trigger
	$("#pcList"+parent+myendtag).trigger("change");
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
        if (matrixInformation[parent].symm) {
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
	populateKspList("kspList"+parent+myendtag,null,"null");
        populatePcList("pcList"+parent+myendtag,null,"null");

	//set defaults for bjacobi
	$("#kspList"+parent+myendtag).find("option[value='preonly']").attr("selected","selected");
        if (matrixInformation[parent].symm) {
            $("#pcList"+parent+myendtag).find("option[value='icc']").attr("selected","selected");
        } else {
	    $("#pcList"+parent+myendtag).find("option[value='ilu']").attr("selected","selected");
        }

        // if parentDiv = bjacobi, it is a Hierarchical Krylov method, display an image for illustration
        var parentDiv_str = parentDiv.substring(0,7);
        if (parentDiv_str == pcValue) {
            alert("parentDiv_str "+ parentDiv_str +" = pcValue, Hierarchical Krylov Method - display an image of the sovler!");
        }
    }
    else {//not bjacobi - why generate, then remove???
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
        if (matrixInformation[parent].symm) {
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
	/*$("#o" + parent).append("<div id=\"ksp"+parent+"\"></div>");*/
	var newDiv = generateDivName(this.id,parent,"ksp");
	var endtag = newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:50px;'></div>");

	//text input box
        var myendtag = endtag+"0";
	$("#"+newDiv).append("<b>KSP KSP   </b><select class=\"kspLists\" id=\"kspList" + parent +myendtag+"\"></select>");
	$("#"+newDiv).append("<br><b>KSP PC &nbsp;&nbsp; </b><select class=\"pcLists\" id=\"pcList" + parent +myendtag+"\"></select>");
	populateKspList("kspList"+parent+myendtag,null,"null");
	populatePcList("pcList"+parent+myendtag,null,"null");

	//set defaults for ksp
	$("#kspList"+parent+myendtag).find("option[value='gmres']").attr("selected","selected");
	$("#pcList"+parent+myendtag).find("option[value='bjacobi']").attr("selected","selected");
	//bjacobi has extra dropdown menus so manually trigger once
	$("#pcList"+parent+myendtag).trigger("change");

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
        if (!matrixInformation[parent].logstruc) {
            alert("Error: Preconditioner fieldsplit cannot be used for non-logically blocked matrix!");
            //how to throw an error???
        } else {
            //set all of its children to parentFieldSplit = true - shall we only set two direct children???
	    var children = []
	    var numChildren = matGetChildren(parent, maxMatricies, children);

	    //set an element of the pclist of the parents children to true
	    for (var i=0; i< numChildren; i++) {
	        $("#pcList" + children[i]).data("parentFieldSplit", true);
            }
        }

        var newDiv = generateDivName(this.id,parent,"fieldsplit");
	var endtag = newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
        //alert("fieldsplit: newDiv="+newDiv);
	$("#"+this.id).after("<div id=\""+newDiv+"\" style='margin-left:50px;'></div>");
        myendtag = endtag+"0";
	$("#"+newDiv).append("<b>Fieldsplit Type &nbsp;&nbsp;</b><select class=\"fieldsplitList\" id=\"fieldsplitList" + parent +myendtag+"\"></select>")  
        populateFieldsplitList("fieldsplitList"+parent+myendtag,null,"null");

    } else { //not fieldsplit 
	var newDiv = generateDivName(this.id,parent,"fieldsplit");
	$("#"+newDiv).remove();
    }

    // if parentFieldSplit is false, disable kspList and pcList 
    for (var i=0; i<maxMatricies; i++) {
	if (!$("#pcList" + i).data("parentFieldSplit")) {
	    $("#pcList" + i).attr("disabled", true)
	    $("#kspList" + i).attr("disabled", true)
	} else {
	    $("#pcList" + i).attr("disabled", false)
	    $("#kspList" + i).attr("disabled", false)
	}
    } 
});

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
	
    if (id.indexOf("_") == -1) { //o div -- fix afer we replace pcList0 with pcList0_???
        newDiv = pcValue + matRecursion + "_";
    } else {
        newDiv = pcValue + matRecursion + endtag;
    } 
    //alert("newDiv "+newDiv);
    return newDiv;
}

/*
  matGetChildren - get all of children of the parent
  input:
    parent - recursionCounter of the parent matrix
    maxMatricies
  output:
    children - array holding the recursionCounter of chidren
    numChildren - number of children
*/
function matGetChildren(parent, maxMatricies, children)
{
    var numChildren = 0;
    if (2 * parent + 2 > maxMatricies) return numChildren;

    //think of a way to code this in an iteration calculate the direct children
    children[0] = 2 * parent + 1;
    children[1] = 2 * parent + 2;
    numChildren += 2;

    //calculate the rest of the children
    var i = 0;
    while (2*children[i] + 2 < maxMatricies) {
	children[numChildren++] = 2 * children[i] + 1
	children[numChildren++] = 2 * children[i] + 2;
        i++;
    }
    return numChildren;
}

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

    //new way of finding parent (the number after o in the o div):  (parent=recursion counter)
    var parentDiv = $(this).parent().get(0).id;
    while (parentDiv.indexOf('_') != -1)
	parentDiv=$("#"+parentDiv).parent().get(0).id;
    var recursionCounter = parentDiv.substring(1, parentDiv.length); //same as recursionCounter. will work when there is more than 1 digit after 'o'

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

