//Make an array. Each array element (matInfo[0]. etc) will have all of the information of the questions. Each array element represents one part of the matrix. [0] will be the big array, [1] will be the second recursion, etc, etc. All the information is stored in the array and accessed via for symmetric matInfo[recursion].symm
var matInfo = [];
var matInfoWriteCounter = 0;//next available space to write to.
var currentAsk = "0";//start at id=0. then 00 01, then 000 001 010 011 etc if splitting two every time.
var askedA0 = false;//a one-way flag to record if A0 was asked

//Used to refer to whether the left top block should be highighted or the right bottom bloc
var matrixPicFlag = 0;

//preRecursionCounter is used to remember the previous counter;
var preRecursionCounter = -1;

//counter of SAWs recursions for '-pc_type'
var recursionCounterSAWs = 0;
var currentRecursionCounterSAWs = 0;
var sawsInfo = [];

//Counter for creating the new divs for the tree
var matDivCounter = 0;

//Use for pcmg
var highestMg       = 0;  //highest mg level encountered so far
var mgLevelLocation = -1; //where to put the mg level data once the highest level is determined. -1 means not yet recorded

//Call the "Tex" function which populates an array with TeX to be used instead of images
//var texMatrices = tex(maxMatricies) //unfortunately, cannot use anymore

//GetAndDisplayDirectory: modified from PETSc.getAndDisplayDirectory 
//------------------------------------------------------------------
GetAndDisplayDirectory = function(names,divEntry){
    //alert("1_start. GetAndDisplayDirectory: name="+name+"; divEntry="+divEntry+"; recursionCounterSAWs="+recursionCounterSAWs);
    jQuery(divEntry).html(""); //Get the HTML contents of the first element in the set of matched elements
    SAWs.getDirectory(names,DisplayDirectory,divEntry);
    //alert("1_end. recursionCounterSAWs "+recursionCounterSAWs);
}

//DisplayDirectory: modified from PETSc.displayDirectory
//------------------------------------------------------
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
        //var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.prefix.data[0];
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["prefix"].data[0];//more accurate I believe
        //alert("saws prefix:"+SAWs_prefix);
        
        if (SAWs_prefix == "(null)") SAWs_prefix = ""; //"(null)" fails populatePcList(), don't know why???
        //create <select> "pcList-1"+SAWs_prefix+" when it is not defined ???
        //$("#pcList-1"+SAWs_prefix).remove();

        if (typeof $("#pcList-1"+SAWs_prefix+"text").attr("title") == "undefined" && SAWs_prefix.indexOf("est")==-1) {//it doesn't exist already and doesn't contain 'est'
            $("#o-1").append("<br><b style='margin-left:20px;' title=\"Preconditioner\" id=\"pcList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"pc_type &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList-1"+SAWs_prefix+"\"></select>");
            populatePcList("pcList-1"+SAWs_prefix,SAWs_alternatives,SAWs_pcVal);
        }
        //alert("Preconditioner (PC) options, SAWs_pcVal "+SAWs_pcVal+", SAWs_prefix "+SAWs_prefix);

        if(SAWs_pcVal == 'mg' && mgLevelLocation==-1)
            mgLevelLocation=recursionCounterSAWs;

        var SAWs_mgLevels="";
        if(SAWs_prefix.indexOf("levels")!=-1) {
            SAWs_mgLevels=SAWs_prefix.substring(10,12);//position 10 and 11. mg levels might be 2 digits long (e.g. greater than 9)

            if(SAWs_mgLevels.indexOf('_')>0)
                SAWs_mgLevels=SAWs_mgLevels.charAt(0);//mg levels is only 1 digit long. remove the extra '_'

            if(SAWs_mgLevels > highestMg)
                highestMg=SAWs_mgLevels;
        }

        var SAWs_bjacobi_blocks="";
        if (SAWs_pcVal == 'bjacobi') {
            
            SAWs_bjacobi_blocks = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-pc_bjacobi_blocks"].data[0];

            //else if(SAWs_prefix == "sub_")...

            //alert("SAWs_bjacobi_blocks "+SAWs_bjacobi_blocks);
            //set SAWs_bjacobi_blocks to #bjacobiBlocks-1_0.processorInput ???
        }

        sawsInfo[recursionCounterSAWs] = {
            prefix: SAWs_prefix,
            bjacobi_blocks: SAWs_bjacobi_blocks
        }
        
        if(mgLevelLocation!=-1)
            sawsInfo[mgLevelLocation].mg_levels=parseInt(highestMg)+1;//need to add 1

        recursionCounterSAWs++;
        //alert("pc: recursionCounterSAWs "+recursionCounterSAWs);

    } else if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Krylov Method (KSP) options") {
        var SAWs_kspVal = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].data[0];
        var SAWs_alternatives = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["-ksp_type"].alternatives;
        var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.prefix.data[0];
        
        if (SAWs_prefix == "(null)") SAWs_prefix = "";
        //$("#kspList-1"+SAWs_prefix).remove();
        if (typeof $("#kspList-1"+SAWs_prefix+"text").attr("title") == "undefined" && SAWs_prefix.indexOf("est")==-1) {//it doesn't exist already and doesn't contain 'est'
            $("#o-1").append("<br><b style='margin-left:20px;' title=\"Krylov method\" id=\"kspList-1"+SAWs_prefix+"text\">-"+SAWs_prefix+"ksp_type &nbsp;</b><select class=\"kspLists\" id=\"kspList-1"+SAWs_prefix+"\"></select>");//giving an html element a title creates a tooltip
            populateKspList("kspList-1"+SAWs_prefix,SAWs_alternatives,SAWs_kspVal);
        }
        //alert("populateKspList is done, SAWs_kspVal "+SAWs_kspVal+", SAWs_prefix "+SAWs_prefix);
    }

    //alert('call SAWs.displayDirectoryRecursive...');
    SAWs.displayDirectoryRecursive(sub.directories,divEntry,0,"");   
}

//When pcoptions.html is loaded ...
//--------------------------------
HandlePCOptions = function(){

    preRecursionCounter = "-1";//A matricies have string id's unlike the previous numerical recursionCounter

    //reset the form
    formSet(currentAsk);

    //hide at first
    $("#fieldsplitBlocks_text").hide();
    $("#fieldsplitBlocks").hide();

    //must define these parameters before setting default pcVal, see populatePcList() and listLogic.js!
    matInfo[-1] = {
        posdef:  false,
        symm:    false,
        logstruc:false,
        blocks:  0,
        matLevel:0,
        id:      "0"
    }

    matInfo[0] = {//surtai added
        posdef:  false,
        symm:    false,
        logstruc:false,
        blocks:  0,
        matLevel:0,
        id:      "0"
    }

    //create div 'o-1' for displaying SAWs options
    $("#divPc").append("<div id=\"o-1\"> </div>");

    // get and display SAWs options
    recursionCounterSAWs = 0;
    GetAndDisplayDirectory("","#variablesInfo");
    //alert("after GetAndDisplayDirectory, recursionCounterSAWs "+recursionCounterSAWs);

    $("#continueButton").click(function(){
        //alert("recursionCounterSAWs "+recursionCounterSAWs+"; prefix="+sawsInfo[0].prefix+" "+sawsInfo[recursionCounterSAWs-1].prefix);

        //todo: DOUBLE CHECK IF VALID INPUT IS PROVIDED. IF NOT, DO NOT CONTINUE

	//matrixLevel is how many matrices deep the data is. 0 is the overall matrix,
        var matrixLevel = currentAsk.length-1;//minus one because A0 is length 1 but level 0
        var fieldsplitBlocks = $("#fieldsplitBlocks").val();

        if (!document.getElementById("logstruc").checked)
            fieldsplitBlocks=0;//sometimes will be left over garbage value from previous submits

	//Write the form data to matInfo
	matInfo[matInfoWriteCounter] = {
            posdef:  document.getElementById("posdef").checked,
            symm:    document.getElementById("symm").checked,
            logstruc:document.getElementById("logstruc").checked,
            blocks:  fieldsplitBlocks,
            matLevel:matrixLevel,
            id:      currentAsk
	}

        //increment write counter immediately after data is written
        matInfoWriteCounter++;

        //append to table of two columns holding A and oCmdOptions in each column (should now be changed to simply cmdOptions)
        //tooltip contains all information previously in big letter format (e.g posdef, symm, logstruc, etc)
        var indentation=matrixLevel*30; //according to the length of currentAsk (aka matrix level), add margins of 30 pixels accordingly
        $("#oContainer").append("<tr> <td> <div style=\"margin-left:"+indentation+"px;\" id=\"A"+ currentAsk + "\" title=\"A"+ currentAsk + " Symm:"+matInfo[matInfoWriteCounter-1].symm+" Posdef:"+matInfo[matInfoWriteCounter-1].posdef+" Logstruc:"+matInfo[matInfoWriteCounter-1].logstruc+"\"> </div></td> <td> <div id=\"oCmdOptions" + currentAsk + "\"></div> </td> </tr>");

        //Create drop-down lists. '&nbsp;' indicates a space
        $("#A" + currentAsk).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>A" + currentAsk +" </b>");
	$("#A" + currentAsk).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>KSP &nbsp;</b><select class=\"kspLists\" id=\"kspList" + currentAsk +"\"></select>");
	$("#A" + currentAsk).append("<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>PC &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList" + currentAsk +"\"></select>");

	//store the recursion counter in the div as a data() - for solverTree and referenced occasionally in listLogic.js although there are other ways to do this
	//$("#kspList" + currentAsk).data("listRecursionCounter", currentAsk);  HAD THIS BEFORE. WILL ADDRESS THIS LATER
	//$("#pcList" + currentAsk).data("listRecursionCounter", currentAsk);
        //set parentFieldSplit:true as default - ugly???
	//$("#pcList" + currentAsk).data("parentFieldSplit",true);

	//populate the kspList and pclist with default options
        if (currentAsk == "0") { //use SAWs options
            var SAWs_kspVal = $("#kspList-1").val();
            //SAWs_alternatives ???
            populateKspList("kspList"+currentAsk,null,SAWs_kspVal);

            var SAWs_pcVal = $("#pcList-1").val(); //Get pctype from the drop-down pcList-1
            //SAWs_alternatives ???
	    populatePcList("pcList"+currentAsk,null,SAWs_pcVal);
            currentRecursionCounterSAWs = 1;
        } else {
            populateKspList("kspList"+currentAsk,null,"null");
            populatePcList("pcList"+currentAsk,null,"null");
        }

        //manually trigger pclist once because additional options, e.g., detailed info may need to be added
	$("#pcList"+currentAsk).trigger("change");

        preRecursionCounter = currentAsk; //save the current counter

        currentAsk = matTreeGetNextNode(currentAsk);
        //alert("new current ask:"+currentAsk);

        formSet(currentAsk); //reset the form

	//Tell mathJax to re compile the tex data
	//MathJax.Hub.Queue(["Typeset",MathJax.Hub]); //unfortunately, cannot use this anymore
    });
}


//  This function is run when the page is first visited
//-----------------------------------------------------
$(document).ready(function(){
$(function() { //needed for jqueryUI tool tip to override native javascript tooltip
    $(document).tooltip();
});

//When the button "Logically Block Structured" is clicked...
//----------------------------------------------------------
$("#logstruc").change(function(){
    if (document.getElementById("logstruc").checked) {
        $("#fieldsplitBlocks_text").show();
        $("#fieldsplitBlocks").show();
        //populatePcList("pcList-1",null,"fieldsplit");//HAD THIS ORIGINALLY
    } else {
        $("#fieldsplitBlocks_text").hide();
        $("#fieldsplitBlocks").hide();
    }
});

//this is ONLY for the input box in the beginning form. NOT the inputs in the A divs (those have class='fieldsplitBlocks')
//-------------------------------------------------------------------------------------------------------------------------
$(document).on("keyup", '.fieldsplitBlocksInput', function() {//alerts user with a tooltip when an invalid input is provided
    if ($(this).val().match(/[^0-9]/) || $(this).val()==0 || $(this).val()==1) {//problem is that integer only bubble still displays when nothing is entered
	$(this).attr("title","hello");//set a random title (this will be overwritten)
	$(this).tooltip();//create a tooltip from jquery UI
	$(this).tooltip({content: "At least 2 blocks!"});//edit displayed text
	$(this).tooltip("open");//manually open once
    } else {
	$(this).removeAttr("title");//remove title attribute
	$(this).tooltip("destroy");
    }
});

//Only show positive definite if symmetric
//----------------------------------------
$("#symm").change(function(){
    if (document.getElementById("symm").checked) {
        $("#posdefRow").show();
    } else {
        $("#posdefRow").hide();
        $("#posdef").removeAttr("checked");
    }
});
    HandlePCOptions();//big function is called here
});

//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------

/*
  formSet - Set Form (hide form if needed)
  input:
    currentAsk
  ouput:
    Form asking questions for currentAsk
*/
function formSet(current)//-1 input for current means that program has finished
{
    if (current=="-1") {
        $("#questions").hide();
        return;
    }

    $("#currentAskText").html("<b id='currentAskText'>Currently Asking for Matrix A"+current+"</b>");
    $("#posdefRow").hide();
    $("#fieldsplitBlocks").hide();
    $("#fieldsplitBlocks_text").hide();
    $("#symm").removeAttr("checked");
    $("#posdef").removeAttr("checked");
    $("#logstruc").removeAttr("checked");

    if(current == "0")//special case for first node since no defaults were set yet
        return;

    //fill in the form according to defaults set by matTreeGetNextNode
    if (matInfo[getMatIndex(current)].symm) {//if symmetric
        $("#posdefRow").show();
        $("#symm").prop("checked", "true");
    }
    //if posdef, fill in bubble
    if (matInfo[getMatIndex(current)].posdef) {
        $("#posdef").prop("checked", "true");
    }
}

/*
  pcGetDetailedInfo - get detailed information from the pclists
  input:
    pcListID
    prefix - prefix of the options in the solverTree
    recursionCounter
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
                var index=getMatIndex(recursionCounter);
                matInfo[index].string      += info;
                matInfo[index].stringshort += infoshort;

                pcGetDetailedInfo(childID,myprefix,recursionCounter,matInfo);
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
    var index=getMatIndex(recursionCounter);
    matInfo[index].string      += info;
    matInfo[index].stringshort += infoshort;

    //is a composite pc so there will be a div in the next position
    var generatedDiv="";
    generatedDiv = $("#"+pcListID).next().get(0).id; //this will be a div, eg. mg0_, bjacobi1_
    //alert("generatedDiv "+generatedDiv);
    for (var i=0; i<$("#"+generatedDiv).children().length; i++) { //loop over all pcLists under this Div
	var childID = $("#"+generatedDiv).children().get(i).id;
	if ($("#"+childID).is(".pcLists")) {//has more pc lists that need to be taken care of recursively
            pcGetDetailedInfo(childID,prefix,recursionCounter,matInfo);
	}
    }
}

/*
  matTreeGetNextNode - uses matInfo to find and return the id of the next node to ask about SKIP ANY CHILDREN FROM NON-LOG STRUC PARENT
  input: 
    currentAsk
  output: 
    id of the next node that should be asked
*/
function matTreeGetNextNode(current)
{
    //important to remember that writeCounter is already pointing at an empty space at this point. this method also initializes the next object if needed.
    if (current=="0" && askedA0)
        return -1;//sort of base case. this only occurs when the tree has completely finished

    if (current=="0")
        askedA0 = true;

    var parentID  = current.substring(0,current.length-1);//simply knock off the last digit of the id
    var lastDigit = current.charAt(current.length-1);
    lastDigit     = parseInt(lastDigit);

    var currentBlocks = matInfo[getMatIndex(current)].blocks;
    var possibleChild = current+""+(currentBlocks-1);

    //case 1: current node needs more child nodes
    if (matInfo[getMatIndex(current)].logstruc && currentBlocks!=0 && getMatIndex(possibleChild)==-1) {//CHECK TO MAKE SURE CHILDREN DON'T ALREADY EXIST
        alert("needs more children");
        matInfo[matInfoWriteCounter]        = new Object();
        matInfo[matInfoWriteCounter].symm   = matInfo[getMatIndex(current)].symm;//set defaults for the new node
        matInfo[matInfoWriteCounter].posdef = matInfo[getMatIndex(current)].posdef;
        return current+"0";//move onto first child
    }

    //case 2: current node's child nodes completed. move on to sister nodes if any
    if (current!="0" && lastDigit+1 < matInfo[getMatIndex(parentID)].blocks) {
        alert("needs more sister nodes");
        matInfo[matInfoWriteCounter]        = new Object();
        matInfo[matInfoWriteCounter].symm   = matInfo[getMatIndex(current)].symm;//set defaults for the new node
        matInfo[matInfoWriteCounter].posdef = matInfo[getMatIndex(current)].posdef;
        var newEnding                       = parseInt(lastDigit)+1;
        return ""+parentID+newEnding;
    }

    if (parentID=="")//only happens when there is only one A matrix
        return -1;

    //case 3: recursive case. both current node's child nodes and sister nodes completed. recursive search starting on parent again
    return matTreeGetNextNode(parentID);
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

    for (var i = 0; i<maxMatricies; i++) {
	if (typeof matInfo[i] != 'undefined') {
	    //get the ksp and pc options at the topest solver-level
	    kspSelectedValue = $("#kspList" + i).val();
	    pcSelectedValue  = $("#pcList" + i).val();

            //get prefix 
            prefix = "-";
            // for pc=fieldsplit
            for (level=1; level<=matInfo[i].matLevel; level++) {
                if (level == matInfo[i].matLevel) {
                    prefix += "fieldsplit_A"+i+"_"; // matInfo[i].name
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
            matInfo[i].string = ("\\(" + matInfo[i].name + "\\) <br /> "+prefix+"ksp_type " + kspSelectedValue + "<br />"+prefix+"pc_type " + pcSelectedValue);
            matInfo[i].stringshort = ("\\(" + matInfo[i].name + "\\) <br /> KSP: " + kspSelectedValue + "; PC: " + pcSelectedValue);

            // for composite pc, get additional info from the rest of pcLists
            pcGetDetailedInfo("pcList"+ i,prefix,i,matInfo);
	}
    }
}

/*
  getMatIndex - 
  input: 
    desired id in string format. (for example, "01001")
  output: 
    index in matInfo where information on that id is located
*/
function getMatIndex(id)
{
    for (var i=0; i<matInfoWriteCounter; i++) {
        if (matInfo[i].id == id)
            return i;//return index where information is located.
    }
    return -1;//invalid id.
}