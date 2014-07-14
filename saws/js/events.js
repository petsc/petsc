//IMPORTANT: always use document.on() because this will also apply to future elements added to document

$(document).on("change","input[id^='logstruc']",function(){
    var id      = this.id;
    var endtag  = id.substring(id.indexOf("0"),id.length);
    var index   = getIndex(matInfo,endtag);
    var checked = $(this).prop("checked");

    matInfo[index].logstruc = checked;
    setDefaults(endtag);
});

$(document).on("change","input[id^='symm']",function(){//blur (disable) posdef to false if symm isn't selected!!
    var id      = this.id;
    var endtag  = id.substring(id.indexOf("0"),id.length);
    var index   = getIndex(matInfo,endtag);
    var checked = $(this).prop("checked");

    matInfo[index].symm = checked;

    if(!checked) { //if not symm, then cannot be posdef
        $("#posdef" + endtag).attr("checked",false);
        $("#posdef" + endtag).attr("disabled", true);
        matInfo[index].posdef = false;
    }
    if(checked) { //if symm, then allow user to adjust posdef
        $("#posdef" + endtag).attr("disabled", false);
    }
    setDefaults(endtag);

    //if symm is checked, disable all child symms ?? (hold off on this)
    //if symm is unchecked, enable all child symms ??
});

$(document).on("change","input[id^='posdef']",function(){
    var id      = this.id;
    var endtag  = id.substring(id.indexOf("0"),id.length);
    var index   = getIndex(matInfo,endtag);
    var checked = $(this).prop("checked");

    matInfo[index].posdef = checked;
    setDefaults(endtag);

    //if posdef is checked, disable all child posdefs ?? (hold off on this)
    //if posdef is unchecked, enable all the posdefs ??
});

//this function is called after a change in symm,posdef,OR logstruc
function setDefaults(endtag) {
    var symm     = $("#symm" + endtag).prop("checked");
    var posdef   = $("#posdef" + endtag).prop("checked");
    var logstruc = $("#logstruc" + endtag).prop("checked");

    var defaults;
    if(endtag == "0") { //has no parent
        defaults = getDefaults("",symm,posdef,logstruc);
    }
    else { //otherwise, we should generate a more appropriate default
        var parentIndex     = getParentIndex(matInfo,endtag);
        var parent_pc_type  = matInfo[parentIndex].pc_type;
        var parent_symm     = matInfo[parentIndex].symm;
        var parent_posdef   = matInfo[parentIndex].posdef;
        var parent_logstruc = matInfo[parentIndex].logstruc;
        defaults            = getDefaults(parent_pc_type,parent_symm,parent_posdef,parent_logstruc,symm,posdef,logstruc); //if this current solver is a sub-solver, then we should set defaults according to its parent. this suggestion will be better.
        var sub_pc_type   = defaults.sub_pc_type;
        var sub_ksp_type  = defaults.sub_ksp_type;
        defaults = {
            pc_type: sub_pc_type,
            ksp_type: sub_ksp_type
        };
    }

    $("#pc_type" + endtag).find("option[value=\"" + defaults.pc_type + "\"]").attr("selected","selected");
    $("#ksp_type" + endtag).find("option[value=\"" + defaults.ksp_type + "\"]").attr("selected","selected");
    //trigger both to add additional options
    $("#ksp_type" + endtag).trigger("change");
    $("#pc_type" + endtag).trigger("change");
}

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

    /*    $("#matrixPic").html("<center>" + "\\(" + getMatrixTex("0") + "\\)" + "</center>");
    if(currentAsk == "-1" && matInfo.length == 1) //no fieldsplits at all, manually add braces
        $("#matrixPic").html("<center>" + "\\(\\left[" + getMatrixTex("0") + "\\right]\\)" + "</center>");
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
});*/

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

