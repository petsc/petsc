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

$(document).on("click","#refresh",function(){
    if(displayCmdOptions) {
        $("#rightPanel").html(""); //clear the rightPanel
        var cmdOptions = getCmdOptions("0","","newline");
        $("#rightPanel").append("<b>Command Line Options:</b><br><br>");
        $("#rightPanel").append(cmdOptions);
    }
    else
        $("#rightPanel").html("");

    if(displayTree) {
       $("#treeContainer").html("<div id='tree'> </div>");
        buildTree();
    }
    else
        $("#treeContainer").html("");

    if(displayMatrix) {
        /*//display matrix pic. manually add square braces the first time
    $("#matrixPic").html("<center>" + "\\(\\left[" + getMatrixTex("0") + "\\right]\\)" + "</center>");
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);*/
         $("#matrixPic").html("<center>" + "\\(" + getMatrixTex("0") + "\\)" + "</center>");
        MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
    }
    else
        $("#matrixPic").html("");

    if(displayDiagram) {
        $("#matrixPic2").html("<center>" + "\\(" + getSpecificMatrixTex("0") + "\\)" + "</center>");
        $("#matrixPic1").html("<center>" + "\\(" + getSpecificMatrixTex2(0) + "\\)" + "</center>");
        MathJax.Hub.Config({ TeX: { extensions: ["AMSMath.js"] }});
        MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
    }
    else {
        $("#matrixPic1").html("");
        $("#matrixPic2").html("");
    }
});

$(document).on("change","#displayCmdOptions",function(){
    displayCmdOptions = $(this).prop("checked");
});

$(document).on("change","#displayTree",function(){
    displayTree = $(this).prop("checked");
});

$(document).on("change","#displayMatrix",function(){
    displayMatrix = $(this).prop("checked");
});

$(document).on("change","#displayDiagram",function(){
    displayDiagram = $(this).prop("checked");
});

/*
//this piece of code is useless right now, but will be restored eventually
$(document).on("keyup", '.processorInput', function() {
    if ($(this).val().match(/[^0-9]/) || $(this).val()<1) {//integer only bubble still displays when nothing is entered
	$(this).attr("title","");//set a random title (this will be overwritten)
	$(this).tooltip();//create a tooltip from jquery UI
	$(this).tooltip({ content: "Integer only!" });//edit displayed text
	$(this).tooltip("open");//manually open once
    } else {
	$(this).removeAttr("title");//remove title attribute
	$(this).tooltip();//create so that we dont call destroy on nothing
        $(this).tooltip("destroy");
    }
});*/
