//IMPORTANT: always use document.on() because this will also apply to future elements added to document

//When "Continue" button is clicked ...
/*$(document).on("click","#continueButton",function(){

    //matrixLevel is how many matrices deep the data is. 0 is the overall matrix,
    var matrixLevel = currentAsk.length-1;//minus one because A0 is length 1 but level 0
    var fieldsplitBlocks = $("#fieldsplitBlocks").val();

    if (!document.getElementById("logstruc").checked)
        fieldsplitBlocks=0;//sometimes will be left over garbage value from previous submits

    //we don't have to worry about possibility of symmetric and not posdef because when symmetric is unchecked, it not only hides posdef but also removes the checkmark if there was one

    //Write the form data to matInfo
    var writeLoc = matInfo.length;
    matInfo[writeLoc] = {
        posdef:  document.getElementById("posdef").checked,
        symm:    document.getElementById("symm").checked,
        logstruc:document.getElementById("logstruc").checked,
        blocks:  fieldsplitBlocks,
        matLevel:matrixLevel,
        id:      currentAsk
    }

    //append to table of two columns holding A and oCmdOptions in each column (should now be changed to simply cmdOptions)
    //tooltip contains all information previously in big letter format (e.g posdef, symm, logstruc, etc)
    var indentation = matrixLevel*30; //according to the length of currentAsk (aka matrix level), add margins of 30 pixels accordingly
    $("#oContainer").append("<tr id='row"+currentAsk+"'> <td> <div style=\"margin-left:"+indentation+"px;\" id=\"A"+ currentAsk + "\"> </div></td> <td> <div id=\"oCmdOptions" + currentAsk + "\"></div> </td> </tr>");

    //Create drop-down lists. '&nbsp;' indicates a space
    $("#A" + currentAsk).append("<br><b id='matrixText"+currentAsk+"'>A" + "<sub>" + currentAsk + "</sub>" + " (Symm:"+matInfo[writeLoc].symm+" Posdef:"+matInfo[writeLoc].posdef+" Logstruc:"+matInfo[writeLoc].logstruc +")</b>");

    $("#A" + currentAsk).append("<br><b>KSP &nbsp;</b><select class=\"kspLists\" id=\"kspList" + currentAsk +"\"></select>");
    $("#A" + currentAsk).append("<br><b>PC &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList" + currentAsk +"\"></select>");

    if(matInfo[writeLoc].logstruc) {//if fieldsplit, need to add the fieldsplit type and fieldsplit blocks
        var newDiv = generateDivName("",currentAsk,"fieldsplit");//this div contains the two fieldsplit dropdown menus. as long as first param doesn't contain "_", it will generate assuming it is directly under an A-div which it is
	var endtag = newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
	$("#A"+currentAsk).append("<div id=\""+newDiv+"\" style='margin-left:"+30+"px;'></div>");//append to the A-div that we just added to the table
        var myendtag = endtag+"0";
	$("#"+newDiv).append("<b>Fieldsplit Type &nbsp;&nbsp;</b><select class=\"fieldsplitList\" id=\"fieldsplitList" + currentAsk +myendtag+"\"></select>");
        $("#"+newDiv).append("<br><b>Fieldsplit Blocks </b><input type='text' id='fieldsplitBlocks"+currentAsk+myendtag+"\' value='"+fieldsplitBlocks+"' maxlength='2' class='fieldsplitBlocks'>");//note that class is fieldsplitBlocks NOT fieldsplitBlocksInput
        populateFieldsplitList("fieldsplitList"+currentAsk+myendtag);
    }

    //populate the kspList and pclist with default options
    populateKspList("kspList"+currentAsk,null,"null");
    populatePcList("pcList"+currentAsk,null,"null");

    //manually trigger pclist once because additional options, e.g., detailed info may need to be added
    if($("#pcList"+currentAsk).val()!="fieldsplit")//but DON'T trigger change on fieldsplit because that would append the required A divs twice
	$("#pcList"+currentAsk).trigger("change");

    currentAsk = matTreeGetNextNode(currentAsk);

    formSet(currentAsk); //reset the form

    $("#matrixPic").html("<center>" + "\\(" + getMatrixTex("0") + "\\)" + "</center>");
    if(currentAsk == "-1" && matInfo.length == 1) //no fieldsplits at all, manually add braces
        $("#matrixPic").html("<center>" + "\\(\\left[" + getMatrixTex("0") + "\\right]\\)" + "</center>");
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
});*/

$(document).on("change","input[id^='logstruc']",function(){//automatically select fieldsplit when logstruc is selected. automatically revert to something else when logstruc is de-selected!!!!!!
    var id     = this.id;
    var endtag = id.substring(id.indexOf("0"),id.length);
    var index  = getIndex(matInfo,endtag);
    var checked = $(this).prop("checked");

    matInfo[endtag].logstruc = checked;

    if(checked && $("#pc_type"+endtag).val() != "fieldsplit") { //if logstruc is selected and is not currently fieldsplit, default to fieldsplit
        $("#pc_type" + endtag).find("option[value='fieldsplit']").attr("selected","selected");
        $("#pc_type" + endtag).trigger("change");
    }
    if(!checked && $("#pc_type"+endtag).val() == "fieldsplit") { //if no longer log struc, then cannot use fieldsplit solver
        $("#pc_type" + endtag).find("option[value='bjacobi']").attr("selected","selected");
        $("#pc_type" + endtag).trigger("change");
    }
});

$(document).on("change","input[id^='symm']",function(){//blur posdef to false if symm isn't selected!!
    var id     = this.id;
    var endtag = id.substring(id.indexOf("0"),id.length);
    var index  = getIndex(matInfo,endtag);
    var checked = $(this).prop("checked");

    matInfo[endtag].symm = checked;

    if(!checked) { //if not symm, then cannot be posdef
        $("#posdef" + endtag).attr("checked",false);
        $("#posdef" + endtag).attr("disabled", true);
    }
    if(checked) { //if symm, then allow user to adjust posdef
        $("#posdef" + endtag).attr("disabled", false);
    }
});

$(document).on("change","input[id^='posdef']",function(){
    var id     = this.id;
    var endtag = id.substring(id.indexOf("0"),id.length);
    var index  = getIndex(matInfo,endtag);
    var checked = $(this).prop("checked");

    matInfo[endtag].posdef = checked;
});

//When "Cmd Options" button is clicked ...
$(document).on("click","#cmdOptionsButton", function(){
    $("#treeContainer").html("<div id='tree'> </div>");

    //build the tree
    buildTree();

    //show cmdOptions to the screen
    $("#rightPanel").html(""); //clear the rightPanel
    var cmdOptions = getCmdOptions("0","","newline");
    $("#rightPanel").append("<b>Command Line Options:</b><br><br>");
    $("#rightPanel").append(cmdOptions);
});

$(document).on("click","#clearOutput",function(){
    $("#rightPanel").html("");
});

$(document).on("click","#clearTree",function(){
    $("#tree").remove();
});

$(document).on("keyup","#selectedMatrix",function(){

    var val = $(this).val();
    if(getMatIndex(val) == -1) //invalid matrix
        return;

    $("#matrixPic2").html("<center>" + "\\(" + getSpecificMatrixTex(val,"") + "\\)" + "</center>");
    $("#matrixPic1").html("<center>" + "\\(" + getSpecificMatrixTex2(0) + "\\)" + "</center>");
    MathJax.Hub.Config({ TeX: { extensions: ["AMSMath.js"] }});
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
});

$(document).on("keyup", '.processorInput', function() {
    if ($(this).val().match(/[^0-9]/) || $(this).val()==0) {//integer only bubble still displays when nothing is entered
	$(this).attr("title","");//set a random title (this will be overwritten)
	$(this).tooltip();//create a tooltip from jquery UI
	$(this).tooltip({ content: "Integer only!" });//edit displayed text
	$(this).tooltip("open");//manually open once
    } else {
	$(this).removeAttr("title");//remove title attribute
	$(this).tooltip();//create so that we dont call destroy on nothing
        $(this).tooltip("destroy");
    }
});

$(document).on("keyup", '.fieldsplitBlocks', function() {//alerts user with a tooltip when an invalid input is provided
    if ($(this).val().match(/[^0-9]/) || $(this).val()==0 || $(this).val()==1) {
	$(this).attr("title","");//set a random title (this will be overwritten)
	$(this).tooltip();//create a tooltip from jquery UI
	$(this).tooltip({content: "At least 2 blocks!"});//edit displayed text
	$(this).tooltip("open");//manually open once
    } else {
	$(this).removeAttr("title");//remove title attribute
        $(this).tooltip();//create so that we dont call destroy on nothing
	$(this).tooltip("destroy");
    }
});

$(document).on("click","#refresh",function(){
    $("#selectedMatrix").trigger("keyup");
});

$(document).on("click","#toggleMatrix",function(){
    if($("#toggleMatrix").val() == "Hide Matrix") {
        $("#matrixPic").hide();
        $("#toggleMatrix").val("Show Matrix");
    }
    else {
        $("#matrixPic").show();
        $("#toggleMatrix").val("Hide Matrix");
    }
});

$(document).on("click","#toggleDiagram",function(){
    if($("#toggleDiagram").val() == "Hide Diagram") {
        $("#matrixPic1").hide();
        $("#matrixPic2").hide();
        $("#toggleDiagram").val("Show Diagram");
    }
    else {
        $("#matrixPic1").show();
        $("#matrixPic2").show();
        $("#toggleDiagram").val("Hide Diagram");
    }
});

