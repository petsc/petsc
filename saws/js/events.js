//IMPORTANT: always use document.on() because this will also apply to future elements added to document

$(document).on("change","input[id^='logstruc']",function(){
    var id      = this.id;
    var endtag  = id.substring(id.indexOf("0"),id.length);
    var checked = $(this).prop("checked");

    matInfo[endtag].logstruc = checked;
    setDefaults(endtag);
});

$(document).on("change","input[id^='symm']",function(){//blur (disable) posdef to false if symm isn't selected!!
    var id      = this.id;
    var endtag  = id.substring(id.indexOf("0"),id.length);
    var checked = $(this).prop("checked");

    matInfo[endtag].symm = checked;

    if(!checked) { //if not symm, then cannot be posdef
        $("#posdef" + endtag).attr("checked",false);
        $("#posdef" + endtag).attr("disabled", true);
        matInfo[endtag].posdef = false;
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
    var checked = $(this).prop("checked");

    matInfo[endtag].posdef = checked;
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
        var parentEndtag    = getParent(endtag);
        var parent_pc_type  = matInfo[parentEndtag].pc_type;
        var parent_symm     = matInfo[parentEndtag].symm;
        var parent_posdef   = matInfo[parentEndtag].posdef;
        var parent_logstruc = matInfo[parentEndtag].logstruc;
        defaults            = getDefaults(parent_pc_type,parent_symm,parent_posdef,parent_logstruc,symm,posdef,logstruc); //if this current solver is a sub-solver, then we should set defaults according to its parent. this suggestion will be better.
        var sub_pc_type   = defaults.sub_pc_type;
        var sub_ksp_type  = defaults.sub_ksp_type;
        defaults = {
            pc_type: sub_pc_type,
            ksp_type: sub_ksp_type
        };
    }

    $("#pc_type" + endtag).val(defaults.pc_type);
    $("#ksp_type" + endtag).val(defaults.ksp_type);
    //trigger both to add additional options
    $("#ksp_type" + endtag).trigger("change");
    $("#pc_type" + endtag).trigger("change");
}

$(document).on("click","#copyToClipboard",function(){
    window.prompt("Copy to clipboard: Ctrl+C, Enter", clipboardText);
    //Note: Because of security reasons, copying to clipboard on click is actually quite complicated. This is a much simpler way to get the job done.
});

//call this method to refresh all the diagram (only the ones that are checked will be displayed)
function refresh() {
        $("#rightPanel").html(""); //clear the rightPanel
        var cmdOptions = getCmdOptions(matInfo,"0","","newline");
        clipboardText  = getCmdOptions(matInfo,"0","","space");
        $("#rightPanel").append("<b>Command Line Options:</b>");
        $("#rightPanel").append("<input type=button value=\"Copy to Clipboard\" id=\"copyToClipboard\" style=\"float:right;font-size:12px;\"><br><br>");
        $("#rightPanel").append(cmdOptions);

        calculateSizes(matInfo,"0");
        var svgString = getBoxTree(matInfo,"0",0,0);
        $("#tree").html("<svg id=\"treeCanvas\" width=\"" + matInfo["0"].total_size.width + "\" height=\"" + matInfo["0"].total_size.height + "\" viewBox=\"0 0 " + matInfo["0"].total_size.width + " " + matInfo["0"].total_size.height + "\">" + svgString + "</svg>");

        $("#matrixPic").html("");

    /*if(displayDiagram) {
        $("#matrixPic2").html("<center>" + "\\(" + getSpecificMatrixTex(matInfo, "0") + "\\)" + "</center>");
        $("#matrixPic1").html("<center>" + "\\(" + getSpecificMatrixTex2(0) + "\\)" + "</center>");
        MathJax.Hub.Config({ TeX: { extensions: ["AMSMath.js"] }});
        MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
    }
    else {
        $("#matrixPic1").html("");
        $("#matrixPic2").html("");
    }*/
}

$(document).on("change","#displayCmdOptions",function(){
    displayCmdOptions = $(this).prop("checked");
    refresh();
});

$(document).on("change","#displayTree",function(){
    displayTree = $(this).prop("checked");
    refresh();
});

$(document).on("change","#displayMatrix",function(){
    displayMatrix = $(this).prop("checked");
    refresh();
});

/*$(document).on("change","#displayDiagram",function(){
    displayDiagram = $(this).prop("checked");
    refresh();
});*/

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
