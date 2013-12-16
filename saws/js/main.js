//Make an array. Each array element (matrixInformation[0]. etc) will have all of the information of the questions. Each array element represents one part of the matrix. [0] will be the big array, [1] will be the second recursion, etc, etc. All the information is stored in the array and accessed via for symmetric matrixInformation[recursion].symm
var matrixInformation = [];

//Change the max amount of allowed matricies here
//based on how many levels of blocking there are
var maxMatriciesLevel = 4;
var maxMatricies      = Math.pow(2, maxMatriciesLevel) - 1;

//The matNode (submatrix) gets its value from whether or not the parent is logically structured. If the parent is not logically structured, the value remains false. If the parent is logically structured, the value is changed to true. The program skips any questions on any recursionCounter whose matNode[] is false
var matNode = [];
for (var i = 0; i < maxMatricies; i++) {
    matNode[i] = false;
}
//Set the first value to true to start the program
matNode[0] = true;

//This array turns the abstract counter into a concret name of the matrix element
var nameOfMatrix = [];
matSetName(maxMatricies, nameOfMatrix);

//Used to refer to whether the left top block should be highighted or the right bottom bloc	
var matrixPicFlag = 0;

//currentRecursionCounter is used to remember the current counter so that it can be used
//to set children to symm and posdef if parent is symm and posdef without
//breaking other code
var currentRecursionCounter = 0;
var recursionCounter        = 0;

//Counter for creating the new divs for the tree
var matDivCounter = 0;

//Call the "Tex" function which populates an array with TeX to be used instead of images
//var texMatrices = tex(maxMatricies)

DisplayPCType = function(defl) {
    SAWs.getDirectory("PETSc/Options/-pc_type",function(data,indef){
        if (indef) var def = indef; 
        else       var def = data.directories.Options.variables["-pc_type"].data[0];
        alert("-pc_type "+def);
        var alternatives = data.directories.Options.variables["-pc_type"].alternatives;
        populatePcList("pcList-1",alternatives,def);
        $("#pcList-1").data("listRecursionCounter", -1);

        //manually trigger pclist once because additional options, e.g., detailed info may need to be added
        $("#pcList-1").trigger("change");

        // here it should display all the other PC options available to the PC currently
    },defl)
}

//GetAndDisplayDirectory: modified from PETSc.getAndDisplayDirectory 
GetAndDisplayDirectory = function(names,divEntry){
    //alert("1. GetAndDisplayDirectory: name="+name+"; divEntry="+divEntry);
    jQuery(divEntry).html(""); //Get the HTML contents of the first element in the set of matched elements
    SAWs.getDirectory(names,DisplayDirectory,divEntry)
}

//DisplayDirectory: modified from PETSc.displayDirectory
DisplayDirectory = function(sub,divEntry)
{
    globaldirectory[divEntry] = sub
    //alert("2. DisplayDirectory: sub="+sub+"; divEntry="+divEntry);
    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {
        //alert("3. divEntry="+divEntry);
        //jQuery(divEntry).append("<center><input type=\"button\" value=\"Continue\" id=\"continue\"></center>")
        //jQuery('#continue').on('click', function(){
            //alert("click continue - sub="+sub+"; divEntry="+divEntry);
            SAWs.updateDirectoryFromDisplay(divEntry)
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
            SAWs.postDirectory(sub);
            jQuery(divEntry).html("");
            window.setTimeout(GetAndDisplayDirectory,1000,null,divEntry);
        //})
    }
   
    if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options") {
        var SAWs_pcVal = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_type"].alternatives;
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.prefix.data[0];
        
        if (SAWs_prefix == "(null)") SAWs_prefix = ""; //"(null)" fails populatePcList(), don't know why???
        $("#o-1").append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>-"+SAWs_prefix+"pc_type &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList-1"+SAWs_prefix+"\"></select>");
        populatePcList("pcList-1"+SAWs_prefix,SAWs_alternatives,SAWs_pcVal);
        //alert("Preconditioner (PC) options, SAWs_pcVal "+SAWs_pcVal+", SAWs_prefix "+SAWs_prefix);

        if (SAWs_pcVal == 'bjacobi') {
            var SAWs_bjacobi_blocks = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_bjacobi_blocks"].data[0];
            //alert("SAWs_bjacobi_blocks "+SAWs_bjacobi_blocks);
            //set SAWs_bjacobi_blocks to #bjacobiBlocks-1_0.processorInput ???
        }

    } else if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Krylov Method (KSP) options") {
        var SAWs_kspVal = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].alternatives;
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.prefix.data[0];
        
        if (SAWs_prefix == "(null)") SAWs_prefix = "";
        $("#o-1").append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b title=\"Krylov method\">-"+SAWs_prefix+"ksp_type &nbsp;</b><select class=\"kspLists\" id=\"kspList"+SAWs_prefix+"\"></select>");//giving an html element a title creates a tooltip

        populateKspList("kspList-1"+SAWs_prefix,SAWs_alternatives,SAWs_kspVal);
        //populateKspList("kspList-1"+SAWs_prefix,null,"null");
        //alert("populateKspList is done, SAWs_kspVal "+SAWs_kspVal+", SAWs_prefix "+SAWs_prefix);
    }

    //alert('call SAWs.displayDirectoryRecursive...');
    SAWs.displayDirectoryRecursive(sub.directories,divEntry,0,"")
}

//When pcoptions.html is loaded ...
HandlePCOptions = function(){
    
    recursionCounter = -1;

    //reset the form
    formSet(recursionCounter,matrixInformation);
   
    //must define these parameters before setting default pcVal, see populatePcList() and listLogic.js!
    matrixInformation[recursionCounter] = {
        posdef:  0,
        symm:    0,
        logstruc:0,
    }
   
    //create div 'o-1' for displaying SAWs options
    $("#divPc").append("<div id=\"o"+recursionCounter+"\"> </div>");
    
    // get SAWs options 
    GetAndDisplayDirectory("","#variablesInfo"); //interfere $("#logstruc, #nlogstruc").change(function() ???
    
    //When the button "Logically Block Structured" is clicked...
    $("#logstruc, #nlogstruc").change(function(){ //why still say !logstruc ???
        matrixInformation[recursionCounter] = {
            posdef:  document.getElementById("posdef").checked,
            symm:    document.getElementById("symm").checked,
            logstruc:document.getElementById("logstruc").checked,
        }
        alert('logstruc='+matrixInformation[recursionCounter].logstruc);
        DisplayPCType("fieldsplit"); //why matrixInformation[recursionCounter].logstruc is not input into $(document).on('change', '.pcLists', function()???
    });
    
    recursionCounter++;

    $("#continueButton").click(function(){
        //alert(recursionCounter);

	//matrixLevel is how many matrices deep the data is. 0 is the overall matrix, 
	// 1 would be 4 blocks, 2 would be 10 blocks, 3 would be 20 blocks, etc
	var matrixLevel = matGetLevel(recursionCounter);
	
	//The data from the form input is saved here
	//Also add a flag to the matrixInformation to know if it is on screen or was skipped
	matrixInformation[recursionCounter] = {
            posdef:  document.getElementById("posdef").checked,
            symm:    document.getElementById("symm").checked,
            logstruc:document.getElementById("logstruc").checked,

	    recursCount:recursionCounter,
	    matLevel:   matrixLevel,
	    name:       nameOfMatrix[recursionCounter]
	}

        //Add a node to the matrix block structure tree
        //matDivCounter = matTreeAddNode(matrixLevel, matDivCounter,recursionCounter,nameOfMatrix);

        //append to table of two columns holding o and oCmdOptions in each column
        $("#oContainer").append("<tr> <td> <div id=\"o"+ recursionCounter + "\"> </div></td> <td> <div id=\"oCmdOptions" + recursionCounter + "\"></div> </td> </tr>");
	
	//Writes the results to screen in that generated matrix block element
        $("#o" + (recursionCounter)).append("<br><br> <font size=3 color=\"#B91A1A\"><b>Matrix \\(" + (nameOfMatrix[recursionCounter]) + "\\)</b></font>")
	
	//Writes more results to screen (&nbsp; is a space; they are added for typesetting purposes)
        if (matrixInformation[recursionCounter].symm) {
	    $("#o" + (recursionCounter)).append("<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Symmetric");
	} else $("#o" + (recursionCounter)).append("<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Non Symmetric");

	if (matrixInformation[recursionCounter].posdef) {
	    $("#o" + (recursionCounter)).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Positive Definite");
	} else $("#o" + (recursionCounter)).append("<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Non Positive Definite");

	if (matrixInformation[recursionCounter].logstruc) {
            if (matrixInformation[recursionCounter].matLevel >= maxMatriciesLevel -1) {
                alert("Warning: Logically block structured is not supported at matrix level "+ matrixInformation[recursionCounter].matLevel);
                matrixInformation[recursionCounter].logstruc = false;
                $("#o" + (recursionCounter)).append("<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Not Logically Block Structured ");
            } else {
	        $("#o" + (recursionCounter)).append("<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Logically Block Structured ");
            }
	} else $("#o" + (recursionCounter)).append("<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Not Logically Block Structured ");

        $(function() { //needed for jqueryUI tool tip to override native javascript tooltip
            $( document ).tooltip();
        });

        //Create drop-down lists
	$("#o" + (recursionCounter)).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b title=\"Krylov method\">KSP &nbsp;</b><select class=\"kspLists\" id=\"kspList" + recursionCounter +"\"></select>");//giving an html element a title creates a tooltip
	$("#o" + (recursionCounter)).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>PC &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList" + recursionCounter +"\"></select>");
       
	//store the recursion counter in the div as a data() - for solverTree - (seems never been used)???
	$("#kspList" + recursionCounter).data("listRecursionCounter", recursionCounter);
	$("#pcList" + recursionCounter).data("listRecursionCounter", recursionCounter);
        //set parentFieldSplit:true as default - ugly???
	$("#pcList" + recursionCounter).data("parentFieldSplit",true);

	//populate the kspList[recursionCounter] and pclist[recursionCounter] with default options
        populateKspList("kspList"+recursionCounter,null,"null");
        if (recursionCounter == 0) {
            var pcVal = $("#pcList-1").val(); //Get pctype from the drop-down pcList-1
	    populatePcList("pcList"+recursionCounter,null,pcVal);
        } else {
            populatePcList("pcList"+recursionCounter,null,"null");
        }

        //manually trigger pclist once because additional options, e.g., detailed info may need to be added
	$("#pcList"+recursionCounter).trigger("change"); 

	//If the matrix is logically structured, set its two children to be true, so the children will be processed
	if (matrixInformation[recursionCounter].logstruc == true  && 2*recursionCounter+2 < maxMatricies) {
	    matNode[2 * recursionCounter + 1] = true;
	    matNode[2 * recursionCounter + 2] = true;  
	}

	//save the current counter
	currentRecursionCounter = recursionCounter;
	//move the counter forward
	recursionCounter++;

	//Find next child node - Skip any children from a non-logically structured parent 
        //Note: this routine changes global variables 'recursionCounter' and 'matDivCounter'!
        matTreeGetNextNode(matNode);

        //Assign the children of a parent its inherited qualities (posdef and symm) 
        matTreeSetChildren(currentRecursionCounter,matrixInformation);

        //reset the form
        formSet(recursionCounter,matrixInformation);
 
	//If we are at the max number of matricies, hide the questions. 
	if (recursionCounter == maxMatricies) {
	    $("#matrixPic").html("<center>" + finalTex(matrixInformation, currentRecursionCounter) + "</center>");
	    $("#questions").hide();
	} else { // If we are not at the finish
	    //produce the tex of the next matrix to screen
	    $("#matrixPic").html("<center>" + tex2(matrixInformation, recursionCounter) + "</center>");
	      
	    //Change the question to ask about the specific matrix at hand
	    $("#isYourMatrix").html("<span><b>Is your matrix \\(" + nameOfMatrix[recursionCounter] + "\\)</b ></span>");
	}

	//Tell mathJax to re compile the tex data
	MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
    })

    //Toggles the output on and off
    //-------------------------------------------------------
    $('#matrixButtons').on('click', ".removeBtn", function () {
        var count = $(this).data("recurs");
        $("#o" + count).toggle();
    })

    //Only show positive definite if symmetric 
    //-------------------------------------------------------
    $("#posdefRow").hide();
    $("#symm").change(function(){
	$("#posdefRow").show();
    })
   
}


//  This function is run when the page is first visited
$(document).ready(function(){HandlePCOptions();})

//------------------------------------------------------------------------------
/*
  matLevelComput - Computes which matrix level (how many blocks deep) the data is
  input:
    number - matrix index
  output: 
    matrixLevel - the level that this matrix 
*/
function matGetLevel(number)
{
    number = number + 1;
    var matrixLevel = 0;
    while (number >= Math.pow(2,matrixLevel))
    {
	matrixLevel = matrixLevel + 1;
    }
    matrixLevel = matrixLevel - 1;
    return matrixLevel;
}

/*
  formSet - Set Form 
  input:
    recursionCounter
    matrixInformation
  ouput:
    Form asking questions for matrix[recursionCounter]
*/
function formSet(recursionCounter,matrixInformation)
{
    $("#posdefRow").hide();
    $("#symm").removeAttr("checked");
    $("#nsymm").removeAttr("checked");
    $("#posdef").removeAttr("checked");
    $("#nposdef").removeAttr("checked");
    $("#logstruc").removeAttr("checked");
    $("#nlogstruc").removeAttr("checked");

    //$("#mg").removeAttr("checked");
    //$("#submitFormButton").removeAttr("checked");
    //$("#submitButton").removeAttr("checked");

    //fill in the form if the information was previously set
    //if symmetric, fill in bubble and show posdef
    if (matrixInformation[recursionCounter] != undefined) {
	if (matrixInformation[recursionCounter].symm) {
	    $("#posdefRow").show();
	    $("#symm").prop("checked", "true");
	}
	//if posdef, fill in bubble
	if (matrixInformation[recursionCounter].posdef) {
	    $("#posdef").prop("checked", "true");
	}
    }
}

/*
  matSetName - Name matrix elements 
  input:  
    maxMatricies - max number of matrices
  output: Name of matrix elements are set as
    nameOfMatrix[0]='A', 
    nameOfMatrix[1]='A_{1}', nameOfMatrix[2]='A_{2}', 
    nameOfMatrix[3]='A_{1_{1}}',...,nameOfMatrix[6]='A_{2_{2}}',
    ...
*/
function matSetName(maxMatricies, nameOfMatrix)
{
    //Counter adds '1' or '2' to the A
    var counter = 1;
    //parentCounter is used to make sure that the child always inherits its parents name
    var parentCounter = 1;
    //The first matrix is A; the rest are just added numbers
    nameOfMatrix[0] = 'A';
    for(var i = 1; i<maxMatricies; i++)
    {
        nameOfMatrix[i] = nameOfMatrix[i - parentCounter] + '_{' + String(counter);
        counter = counter + 1;
        if (counter == 2)
	    parentCounter = parentCounter + 1;
        if (counter == 3)
	    counter = 1;   
    }

    //Add the end brackets
    for (var i = 1; i<maxMatricies; i++)
    {
        var matLevel = matGetLevel(i);
        for (var j = 0; j<matLevel; j++)
	    nameOfMatrix[i] = nameOfMatrix[i] + '}';
    }
}

/*
  pcGetDetailedInfo - get detailed information from the pclists
  input:
    pcListID
    prefix - prefix of the options in the solverTree
    recursionCounter
  output:
    matrixInformation.string
    matrixInformation.stringshort
*/
function pcGetDetailedInfo(pcListID, prefix,recursionCounter,matrixInformation) 
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
        matrixInformation[recursionCounter].string      += info;
        matrixInformation[recursionCounter].stringshort += infoshort;

        var smoothingKSP;
        var smoothingPC;
        var coarseKSP;
        var coarsePC;
        var level;

        //is a composite pc so there will be a div in the next position
        var generatedDiv="";
        generatedDiv = $("#"+pcListID).next().get(0).id; //this will be a div
        //alert("generatedDiv "+generatedDiv+"; children().length="+$("#"+generatedDiv).children().length);
        level = mgLevels-1;
        for (var i=0; i<$("#"+generatedDiv).children().length; i++) { //loop over all pcLists under this Div
	    var childID = $("#"+generatedDiv).children().get(i).id;
	    if ($("#"+childID).is(".pcLists")) {//has more pc lists that need to be taken care of recursively
                info      = "";
                infoshort = "";
                if (level) {
		    
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

                matrixInformation[recursionCounter].string      += info;
                matrixInformation[recursionCounter].stringshort += infoshort;

                pcGetDetailedInfo(childID,myprefix,recursionCounter,matrixInformation);
                level--;
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
    matrixInformation[recursionCounter].string      += info;
    matrixInformation[recursionCounter].stringshort += infoshort;

    //is a composite pc so there will be a div in the next position
    var generatedDiv="";
    generatedDiv = $("#"+pcListID).next().get(0).id; //this will be a div, eg. mg0_, bjacobi1_
    //alert("generatedDiv "+generatedDiv);
    for (var i=0; i<$("#"+generatedDiv).children().length; i++) { //loop over all pcLists under this Div
	var childID = $("#"+generatedDiv).children().get(i).id;
	if ($("#"+childID).is(".pcLists")) {//has more pc lists that need to be taken care of recursively
            pcGetDetailedInfo(childID,prefix,recursionCounter,matrixInformation);
	}
    }
}

/*
  matTreeAddNode - add a node to the matrix block structure tree
  input:
    matrixLevel, matDivCounter,recursionCounter,nameOfMatrix
  output:
    matDivCounter
*/
function matTreeAddNode(matrixLevel, matDivCounter,recursionCounter,nameOfMatrix)
{
    //add a line break for each new level
    if (matrixLevel != matDivCounter)
    {
	matDivCounter = matDivCounter+1;
	$("l1").append("<br>");
    }
	
    //Add a button to allow the toggle of the output resuls. The tree is made via style on the .html page. The style just centers the multiple elements across the pane
    $("l1").append("<u1><button class=\"removeBtn\" id=\"recursion" + recursionCounter + "\">" + "\\(" + nameOfMatrix[recursionCounter] + "\\) </button></u1>")
	
    //Attach the recursion number to the html element to allow better inspection of resutls
    $("#recursion" + recursionCounter).data("recurs", recursionCounter);
    return matDivCounter;
}

/*
  matTreeSetChildren - set the children of mat[currentRecursion] its inherited qualities (posdef and symm) 
  input:
    recursionCounter - index of  matrixInformation
    matrixInformation 
  output:
    matrixInformation 
*/
function matTreeSetChildren(recursionCounter, matrixInformation)
{
    if (matrixInformation[recursionCounter].logstruc == true && 2*recursionCounter+2 < maxMatricies) {
        var childleft  = 2 * recursionCounter + 1;
        var childright = 2 * recursionCounter + 2;
	matrixInformation[childleft] = {};
	matrixInformation[childright] = {};

	//if symmetric, all children are symmetric
	if (matrixInformation[recursionCounter].symm) {
	    matrixInformation[childleft].symm = true;
	    matrixInformation[childright].symm = true;
	} else {
            matrixInformation[childleft].symm = false;
	    matrixInformation[childright].symm = false;
        }

	//if positive definite, all children are positive definite
	if (matrixInformation[recursionCounter].posdef) {
	    matrixInformation[childleft].posdef = true;
	    matrixInformation[childright].posdef = true;
	} else {
            matrixInformation[childleft].posdef = false;
	    matrixInformation[childright].posdef = false;
        }
    }
}

/*
  matTreeGetNextNode - Find next child node - Skip any children from a non-logically structured parent
  input:
    matNode
  output:
    global variables 'recursionCounter' and 'matDivCounter' are changed by this function!
*/
function matTreeGetNextNode(matNode)
{
    while (matNode[recursionCounter] == false && recursionCounter < maxMatricies) {   
	recursionCounter += 1;
	//add some blank elements to the tree to give it a better structure
	matrixLevel = matGetLevel(recursionCounter);
	//add a line break for new level
	if (matrixLevel != matDivCounter) {
	    matDivCounter += 1;
	    $("l1").append("<br>");
	}
	$("l1").append("<u1 style=\"width:20px\"></u1>");
    }
}

/*
  solverGetOptions - get the options from the drop-down lists
  input:
    matrixInformation
  output:
    matrixInformation[].string stores collected solver options
    matrixInformation[].stringshort
*/
function solverGetOptions(matrixInformation)
{
    var prefix,kspSelectedValue,pcSelectedValue,level;

    for (var i = 0; i<maxMatricies; i++) {
	if (typeof matrixInformation[i] != 'undefined') {
	    //get the ksp and pc options at the topest solver-level
	    kspSelectedValue = $("#kspList" + i).val();
	    pcSelectedValue  = $("#pcList" + i).val();

            //get prefix 
            prefix = "-";
            // for pc=fieldsplit
            for (level=1; level<=matrixInformation[i].matLevel; level++) {
                if (level == matrixInformation[i].matLevel) {
                    prefix += "fieldsplit_A"+i+"_"; // matrixInformation[i].name
                } else { 
                    var parent = Math.floor((i-1)/2);
                    var parentLevel = matGetLevel(parent);
                    while (level < parentLevel) {
                        parent = Math.floor((parent-1)/2);
                        parentLevel = matGetLevel(parent);
                    }
                    prefix += "fieldsplit_A"+parent+"_"; 
                }
            }

	    //together, with the name, make a full string for printing
            matrixInformation[i].string = ("\\(" + matrixInformation[i].name + "\\) <br /> "+prefix+"ksp_type " + kspSelectedValue + "<br />"+prefix+"pc_type " + pcSelectedValue);
            matrixInformation[i].stringshort = ("\\(" + matrixInformation[i].name + "\\) <br /> KSP: " + kspSelectedValue + "; PC: " + pcSelectedValue);

            // for composite pc, get additional info from the rest of pcLists
            pcGetDetailedInfo("pcList"+ i,prefix,i,matrixInformation);
	}
    }
}
