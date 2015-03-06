//this file contains all the code for the input interface through the tree diagram

//global variables to keep track of where the input box is on the page
var boxPresent = false;
var boxEndtag = "";
var box_x = 0;
var box_y = 0;
var box_size = new Object();

function removeBox(){
    $("#tempInput").remove();
}

//this method submits the options and generates the children (if any) with appropriate defaults
//returns 1 on success and 0 on fail
function submitOptions(){

    var endtag = boxEndtag;
    var parentEndtag = getParent(endtag);

    var selectedPc = $("#temp_pc_type").val();

    var symm;
    var posdef;
    var logstruc;

    if($("#temp_symm").length == 0) { //if the checkbox options were not displayed
        symm = matInfo[endtag].symm;
        posdef = matInfo[endtag].posdef;
        logstruc = matInfo[endtag].logstruc;
    }

    else {
        symm     = $("#temp_symm").prop("checked");
        posdef   = $("#temp_posdef").prop("checked");
        logstruc = $("#temp_logstruc").prop("checked");
    }

    //check if matrix properties are valid
    if(!symm && posdef) {
        alert("Error: Cannot be non-symmetric yet positive definite");
        return 0;
    }

    if(endtag != "0") { //check for discrepancies with parent
        var parentSymm = matInfo[parentEndtag].symm;
        var parentPosdef = matInfo[parentEndtag].posdef;
        var parentLogstruc = matInfo[parentEndtag].logstruc;
        if(parentSymm && !symm) {
            alert("Error: Cannot have symmetric parent yet be non-symmetric");
            return 0;
        }
        if(parentPosdef && !posdef) {
            alert("Error: Cannot have positive definite parent yet be non-positive definite");
            return 0;
        }
    }

    //check for more invalid setups
    if(selectedPc == "fieldsplit") {
        var blocks = $("#temp_pc_fieldsplit_blocks").val();
        if(blocks.match(/[^0-9]/) || blocks < 1) {
            alert("Error: Must have at least 1 block for fieldsplit");
            return 0;
        }
        /*if(!logstruc) {
            alert("Error: Cannot use fieldsplit on non-logically block structured matrix");
            return 0;
        }*/
    }
    else if(selectedPc == "mg") { //extra options for mg
        var levels = $("#temp_pc_mg_levels").val();
        if(levels.match(/[^0-9]/) || levels < 1) {
            alert("Error: Must have at least 1 level for multigrid");
            return 0;
        }
    }
    else if(selectedPc == "gamg") {
        var levels = $("#temp_pc_gamg_levels").val();
        if(levels.match(/[^0-9]/) || levels < 1) {
            alert("Error: Must have at least 1 level for geometric algebraic multigrid");
            return 0;
        }
    }
    else if(selectedPc == "bjacobi") {
        var blocks = $("#temp_pc_bjacobi_blocks").val();
        if(blocks.match(/[^0-9]/) || blocks < 1) {
            alert("Error: Must have at least 1 block for bjacobi");
            return 0;
        }
    }
    else if(selectedPc == "redundant") {
        var number = $("#temp_pc_redundant_number").val();
        if(number.match(/[^0-9]/) || number < 1) {
            alert("Error: Redundant Number must be at least 1");
            return 0;
        }
    }
    else if(selectedPc == "asm") {
        var blocks = $("#temp_pc_asm_blocks").val();
        if(blocks.match(/[^0-9]/) || blocks < 1) {
            alert("Error: Must have at least 1 block for Additive Schwarz Method");
            return 0;
        }
        var overlap = $("#temp_pc_asm_overlap").val();
        if(overlap.match(/[^0-9]/) || overlap < 1) {
            alert("Error: Must have overlap of at least 1 for Additive Schwarz Method");
            return 0;
        }
    }


    //only preserve some options if the pc_type didn't change
    var oldPc = matInfo[endtag].pc_type;
    var newPc = $("#temp_pc_type").val();
    var changedPc = (oldPc != newPc);
    if(changedPc)
        deleteAllChildren(endtag);

    //write options to matInfo
    matInfo[endtag].symm = symm;
    matInfo[endtag].posdef = posdef;
    matInfo[endtag].logstruc = logstruc;

    matInfo[endtag].pc_type = $("#temp_pc_type").val();
    matInfo[endtag].ksp_type = $("#temp_ksp_type").val();

    //update dropdown list interface
    $("#symm" + endtag).prop("checked",symm);
    $("#posdef" + endtag).prop("checked",posdef);
    $("#logstruc" + endtag).prop("checked",logstruc);

    if(!symm)
        $("#posdef" + endtag).attr("disabled", true);
    if(symm)
        $("#posdef" + endtag).attr("disabled", false);

    $("#pc_type" + endtag).val(matInfo[endtag].pc_type);
    $("#ksp_type" + endtag).val(matInfo[endtag].ksp_type);

    var pc_type = matInfo[endtag].pc_type;

    if(pc_type == "fieldsplit") { //extra options for fieldsplit
        var blocks = $("#temp_pc_fieldsplit_blocks").val();
        var type   = $("#temp_pc_fieldsplit_type").val();

        if(changedPc)
            $("#pc_type" + endtag).trigger("change");

        //set the appropriate option and then trigger the change in the field (this writes to matInfo)
        $("#pc_fieldsplit_type" + endtag).val(type);
        $("#pc_fieldsplit_type" + endtag).trigger("change");
        $("#pc_fieldsplit_blocks" + endtag).val(blocks);
        $("#pc_fieldsplit_blocks" + endtag).trigger("keyup");
    }
    else if(pc_type == "mg") { //extra options for mg
        var levels = $("#temp_pc_mg_levels").val();
        var type   = $("#temp_pc_mg_type").val();

        if(changedPc)
            $("#pc_type" + endtag).trigger("change");

        //set the appropriate option and then trigger the change in the field (this writes to matInfo)
        $("#pc_mg_type" + endtag).val(type);
        $("#pc_mg_type" + endtag).trigger("change");
        $("#pc_mg_levels" + endtag).val(levels);
        $("#pc_mg_levels" + endtag).trigger("keyup");
    }
    else if(pc_type == "gamg") {
        var levels = $("#temp_pc_gamg_levels").val();
        var type   = $("#temp_pc_gamg_type").val();

        if(changedPc)
            $("#pc_type" + endtag).trigger("change");

        //set the appropriate option and then trigger the change in the field (this writes to matInfo)
        $("#pc_gamg_type" + endtag).val(type);
        $("#pc_gamg_type" + endtag).trigger("change");
        $("#pc_gamg_levels" + endtag).val(levels);
        $("#pc_gamg_levels" + endtag).trigger("keyup");
    }
    else if(pc_type == "bjacobi") {
        var blocks = $("#temp_pc_bjacobi_blocks").val();
        if(changedPc)
            $("#pc_type" + endtag).trigger("change");
        $("#pc_bjacobi_blocks" + endtag).val(blocks);
        $("#pc_bjacobi_blocks" + endtag).trigger("keyup");
    }
    else if(pc_type == "redundant") {
        var number = $("#temp_pc_redundant_number").val();
        if(changedPc)
            $("#pc_type" + endtag).trigger("change");
        $("#pc_redundant_number" + endtag).val(number);
        $("#pc_redundant_number" + endtag).trigger("keyup");
    }
    else if(pc_type == "asm") {
        var blocks  = $("#temp_pc_asm_blocks").val();
        var overlap = $("#temp_pc_asm_overlap").val();
        if(changedPc)
            $("#pc_type" + endtag).trigger("change");
        $("#pc_asm_blocks" + endtag).val(blocks);
        $("#pc_asm_blocks" + endtag).trigger("keyup");
        $("#pc_asm_overlap" + endtag).val(overlap);
        $("#pc_asm_overlap" + endtag).trigger("keyup");
    }
    else if(pc_type == "ksp") {
        //ksp doesn't have any additional options, but it has one child
        if(changedPc)
            $("#pc_type" + endtag).trigger("change");
    }

    enforceProperties(endtag);

    return 1;
}

// this function enforces the logical properties of the matrix
function enforceProperties(endtag) {

    var symm  = matInfo[endtag].symm;
    var posdef = matInfo[endtag].posdef;

    if(symm) { //if symm, then all children must be symm
        for (var key in matInfo) {
            if (matInfo.hasOwnProperty(key)) {
                if(key.indexOf(endtag) == 0) { //if it is indeed a child
                    matInfo[key].symm = true;
                }
            }
        }
    }

    if(posdef) { //if posdef, then all children must be posdef
        for (var key in matInfo) {
            if (matInfo.hasOwnProperty(key)) {
                if(key.indexOf(endtag) == 0) { //if it is indeed a child
                    matInfo[key].posdef = true;
                }
            }
        }
    }

}

//this function ensures that all children are properly generated and initialized
function generateChildren(endtag) {

    var pc_type = matInfo[endtag].pc_type;

    if(pc_type == "bjacobi") { //needs 1 child
        //check if that 1 child already exists. if so, then stop. otherwise, generate that child, select the defaults, and recursively ensure that the new child also has the appropriate children
        var childEndtag = endtag + "_0";
        if(matInfo[childEndtag] == undefined) { //generate child
            var defaults = getDefaults("bjacobi",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

            matInfo[childEndtag] = {
                pc_type:  defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm:     matInfo[endtag].symm, //inherit !!
                posdef:   matInfo[endtag].posdef,
                logstruc: matInfo[endtag].logstruc
            };
            generateChildren(childEndtag);
        }
    }
    else if(pc_type == "fieldsplit") {
        for(var i=0; i<matInfo[endtag].pc_fieldsplit_blocks; i++) {
            var childEndtag = endtag + "_" + i;
            if(matInfo[childEndtag] == undefined) { //generate child
                var defaults = getDefaults("fieldsplit",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

                matInfo[childEndtag] = {
                    pc_type:  defaults.sub_pc_type,
                    ksp_type: defaults.sub_ksp_type,
                    symm:     matInfo[endtag].symm, //inherit !!
                    posdef:   matInfo[endtag].posdef,
                    logstruc: false //to prevent infinite recursion
                };
                generateChildren(childEndtag);
            }
        }
    }
    else if(pc_type == "mg") {
        for(var i=0; i<matInfo[endtag].pc_mg_levels; i++) {
            var childEndtag = endtag + "_" + i;
            if(matInfo[childEndtag] == undefined) { //generate child
                var defaults = getDefaults("mg",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

                matInfo[childEndtag] = {
                    pc_type:  defaults.sub_pc_type,
                    ksp_type: defaults.sub_ksp_type,
                    symm:     matInfo[endtag].symm, //inherit !!
                    posdef:   matInfo[endtag].posdef,
                    logstruc: matInfo[endtag].logstruc
                };
                generateChildren(childEndtag);
            }
        }
    }
    else if(pc_type == "gamg") {
        for(var i=0; i<matInfo[endtag].pc_gamg_levels; i++) {
            var childEndtag = endtag + "_" + i;
            if(matInfo[childEndtag] == undefined) { //generate child
                var defaults = getDefaults("gamg",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

                matInfo[childEndtag] = {
                    pc_type:  defaults.sub_pc_type,
                    ksp_type: defaults.sub_ksp_type,
                    symm:     matInfo[endtag].symm, //inherit !!
                    posdef:   matInfo[endtag].posdef,
                    logstruc: matInfo[endtag].logstruc
                };
                generateChildren(childEndtag);
            }
        }
    }
    else if(pc_type == "ksp") {
        var childEndtag = endtag + "_0";
        if(matInfo[childEndtag] == undefined) { //generate child
            var defaults = getDefaults("ksp",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

            matInfo[childEndtag] = {
                pc_type:  defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm:     matInfo[endtag].symm, //inherit !!
                posdef:   matInfo[endtag].posdef,
                logstruc: matInfo[endtag].logstruc
            };
            generateChildren(childEndtag);
        }
    }

}

$(document).on("click","input[id='setOptions']",function(){

    var result = submitOptions(); //submitOptions() returns 0 on fail and 1 on success
    if(result == 1) {
        removeBox();
        boxPresent = false;
        refresh();
    }
});

$(document).on("change","input[id='temp_symm']",function(){
    setDefaults2();
});

$(document).on("change","input[id='temp_posdef']",function(){
    setDefaults2();
});

$(document).on("change","input[id='temp_logstruc']",function(){
    setDefaults2();
});

//this function sets default ksp/pc when the properties of the matrix are changed
function setDefaults2() {
    var symm     = $("#temp_symm").prop("checked");
    var posdef   = $("#temp_posdef").prop("checked");
    var logstruc = $("#temp_logstruc").prop("checked");

    var endtag = boxEndtag;

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
        defaults = {
            pc_type: defaults.sub_pc_type,
            ksp_type: defaults.sub_ksp_type
        };
    }

    $("#temp_pc_type").val(defaults.pc_type);
    $("#temp_ksp_type").val(defaults.ksp_type);

    $("#temp_pc_type").trigger("change"); //display the correct options
}

//this function displays the appropriate options for each pc_type when pc_type dropdown is changed (for example, bjacobi blocks, fieldsplit type, etc)
$(document).on("change","select[id='temp_pc_type']", function(){

    var endtag = boxEndtag;
    var pc_type = $(this).val();

    //first, remove all the existing suboptions
    $("#temp_pc_type").nextAll().remove();//remove the options in the same level solver

    if(pc_type == "fieldsplit") { //add the extra options that fieldsplit requires
        appendExtraOptions("fieldsplit");
        var defaults = getDefaults("fieldsplit",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        $("#temp_pc_fieldsplit_type").val(defaults.pc_fieldsplit_type);
        $("#temp_pc_fieldsplit_blocks").val(defaults.pc_fieldsplit_blocks);
    }
    else if(pc_type == "mg") {
        appendExtraOptions("mg");
        var defaults = getDefaults("mg",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        $("#temp_pc_mg_type").val(defaults.pc_mg_type);
        $("#temp_pc_mg_levels").val(defaults.pc_mg_levels);
    }
    else if(pc_type == "bjacobi") {
        appendExtraOptions("bjacobi");
        var defaults = getDefaults("bjacobi",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        $("#temp_pc_bjacobi_blocks").val(defaults.pc_bjacobi_blocks);
    }
    else if(pc_type == "gamg") {
        appendExtraOptions("gamg");
        var defaults = getDefaults("gamg",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        $("#temp_pc_gamg_type").val(defaults.pc_gamg_type);
        $("#temp_pc_gamg_levels").val(defaults.pc_gamg_levels);
    }
    else if(pc_type == "redundant") {
        appendExtraOptions("redundant");
        var defaults = getDefaults("redundant",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        $("#temp_pc_redundant_number").val(defaults.pc_redundant_number);
    }
    else if(pc_type == "asm") {
        appendExtraOptions("asm");
        var defaults = getDefaults("asm",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        $("#temp_pc_asm_blocks").val(defaults.pc_asm_blocks);
        $("#temp_pc_asm_overlap").val(defaults.pc_asm_overlap);
    }

    $("#tempInput").append("<br><input type=\"button\" value=\"Set Options\" id=\"setOptions\">"); //put the button back because we still want that (was removed in the nextAll().remove())
});

//remove the box when the user clicks sufficiently outside of it (still needs major debugging)
/*$(document).on("click", function(e) {

    if(e.pageX < (box_x-node_radius-5) || e.pageX > (box_x + box_size.width) || e.pageY < (box_y-node_radius-5)) {
        //note: the reason we dont have the 'greater than y' term is because the dropdown menu goes below the box
        removeBox();
        //alert('removed');
        //alert(e.pageX < (box_x-node_radius-5));
        //alert(e.pageX > (box_x + box_size.width));
        //alert(e.pageY < (box_y-node_radius-5));
        console.log(e.pageX);
        console.log(e.pageY);
        boxPresent = false;
    }
});*/

//upon a user click, present the user with the currently selected options that the user can change
$(document).on("click","circle[id^='node']",function() {

    var id     = $(this).attr("id");//really should not be used in this method. there are better ways of getting information
    var endtag = id.substring(id.indexOf("0"),id.length);
    var parentEndtag = getParent(endtag);
    //scrollTo("solver"+endtag);
    var x = $(this).attr("cx");
    var y = $(this).attr("cy");

    if(boxPresent && boxEndtag == endtag) { //user clicked the same node again
        removeBox();
        boxPresent = false;
        return;
    }
    else if(boxPresent && boxEndtag != endtag) { //user clicked a different node
        removeBox();
        boxPresent = true;
        boxEndtag = endtag;
    }
    else {
        boxPresent = true;
        boxEndtag = endtag;
    }

    //append an absolute-positioned div to display options for that node
    var svgCanvas = $(this).parent().get(0);
    var parent    = $(svgCanvas).parent().get(0);
    var parent_x = parseFloat($(svgCanvas).offset().left) + parseFloat(x) + node_radius;
    var parent_y = parseFloat($(svgCanvas).offset().top) + parseFloat(y) + node_radius;
    $(parent).append("<div id=\"tempInput\" style=\"z-index:1;position:absolute;left:" + (parent_x) + "px;top:" + (parent_y) + "px;font-size:14px;opacity:1;border:2px solid lightblue;border-radius:" + node_radius + "px;\"></div>");
    $("#tempInput").css("background", "#dddddd");

    box_x = parent_x;
    box_y = parent_y;

    var childNum = endtag.substring(endtag.lastIndexOf("_")+1, endtag.length);

    var solverText = "";
    if(endtag == "0")
        solverText = "<b>Root Solver Options (Matrix is <input type=\"checkbox\" id=\"temp_symm" + "\">symmetric,  <input type=\"checkbox\" id=\"temp_posdef" + "\">positive definite, <input type=\"checkbox\" id=\"temp_logstruc" + "\">block structured)</b>";
    else if(matInfo[parentEndtag].pc_type == "bjacobi")
        solverText = "<b>" + "Bjacobi Solver Options" + "</b>";
    else if(matInfo[parentEndtag].pc_type == "fieldsplit")
        solverText = "<b>Fieldsplit " + childNum + " Options (Matrix is <input type=\"checkbox\" id=\"temp_symm" + "\">symmetric,  <input type=\"checkbox\" id=\"temp_posdef" + "\">positive definite, <input type=\"checkbox\" id=\"temp_logstruc" + "\">block structured)</b>";
    else if(matInfo[parentEndtag].pc_type == "redundant")
        solverText = "<b>" + "Redundant Solver Options" + "</b>";
    else if(matInfo[parentEndtag].pc_type == "asm")
        solverText = "<b>" + "ASM Solver Options" + "</b>";
    else if(matInfo[parentEndtag].pc_type == "ksp")
        solverText = "<b>" + "KSP Solver Options" + "</b>";
    else if(matInfo[parentEndtag].pc_type == "mg" || matInfo[parentEndtag].pc_type == "gamg") {
        if(childNum == 0) //coarse grid solver (level 0)
            solverText = "<b> Coarse Grid Solver (Level 0)  </b>";
        else
            solverText = "<b>Smoothing (Level " + childNum + ")  </b>";
    }

    $("#tempInput").append(solverText);
    $("#tempInput").append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"temp_ksp_type" + "\"></select>");
    $("#tempInput").append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"temp_pc_type" + "\"></select>");

    populateList("pc",endtag,"#temp_pc_type"); //this is kind of stupid right now. I'll fix this later
    populateList("ksp",endtag,"#temp_ksp_type");
    $("#temp_pc_type").val(matInfo[endtag].pc_type);
    $("#temp_ksp_type").val(matInfo[endtag].ksp_type);

    //fill in the currently selected properties
    if(matInfo[endtag].symm)
        $("#temp_symm").attr("checked", true);
    if(matInfo[endtag].posdef)
        $("#temp_posdef").attr("checked", true);
    if(matInfo[endtag].logstruc)
        $("#temp_logstruc").attr("checked", true);

    if(matInfo[endtag].pc_type == "fieldsplit") { //append extra options for fieldsplit
        appendExtraOptions("fieldsplit");
        $("#temp_pc_fieldsplit_type").val(matInfo[endtag].pc_fieldsplit_type);
        $("#temp_pc_fieldsplit_blocks").val(matInfo[endtag].pc_fieldsplit_blocks);
    }
    else if(matInfo[endtag].pc_type == "bjacobi") { //append extra options for bjacobi
        appendExtraOptions("bjacobi");
        $("#temp_pc_bjacobi_blocks").val(matInfo[endtag].pc_bjacobi_blocks);
    }
    else if(matInfo[endtag].pc_type == "redundant") {
        appendExtraOptions("redundant");
        $("#temp_pc_redundant_number").val(matInfo[endtag].pc_redundant_number);
    }
    else if(matInfo[endtag].pc_type == "asm") {
        appendExtraOptions("asm");
        $("#temp_pc_asm_blocks").val(matInfo[endtag].pc_asm_blocks);
        $("#temp_pc_asm_overlap").val(matInfo[endtag].pc_asm_overlap);
    }
    else if(matInfo[endtag].pc_type == "mg") {
        appendExtraOptions("mg");
        $("#temp_pc_mg_type").val(matInfo[endtag].pc_mg_type);
        $("#temp_pc_mg_levels").val(matInfo[endtag].pc_mg_levels);
    }
    else if(matInfo[endtag].pc_type == "gamg") {
        appendExtraOptions("gamg");
        $("#temp_pc_gamg_type").val(matInfo[endtag].pc_gamg_type);
        $("#temp_pc_gamg_levels").val(matInfo[endtag].pc_gamg_levels);
    }

    //append the submit button
    $("#tempInput").append("<br><input type=\"button\" value=\"Set Options\" id=\"setOptions\">");

    box_size = {
        height: $("#tempInput").height(),
        width: $("#tempInput").width()
    };

});

//this function only appends the appropriate html. it does NOT select the appropriate defaults
function appendExtraOptions(pc_type) {

    if(pc_type == "fieldsplit") { //append extra options for fieldsplit
        $("#tempInput").append("<br><b>Fieldsplit Type &nbsp;&nbsp;</b><select id=\"temp_pc_fieldsplit_type" + "\"></select>");
        $("#tempInput").append("<br><b>Fieldsplit Blocks </b><input type='text' id=\"temp_pc_fieldsplit_blocks" + "\" maxlength='4'>");
        populateList("fieldsplit","", "#temp_pc_fieldsplit_type");
    }
    else if(pc_type == "bjacobi") { //append extra options for bjacobi
        $("#tempInput").append("<br><b>Bjacobi Blocks </b><input type='text' id=\'temp_pc_bjacobi_blocks" + "\' maxlength='4'>");
    }
    else if(pc_type == "redundant") {
        $("#tempInput").append("<br><b>Redundant Number </b><input type='text' id=\'temp_pc_redundant_number" + "\' maxlength='4'>");
    }
    else if(pc_type == "asm") {
        $("#tempInput").append("<br><b>ASM blocks   &nbsp;&nbsp;</b><input type='text' id=\"temp_pc_asm_blocks" + "\" maxlength='4'>");
	$("#tempInput").append("<br><b>ASM overlap   </b><input type='text' id=\"temp_pc_asm_overlap" + "\" maxlength='4'>");
    }
    else if(pc_type == "mg") {
        $("#tempInput").append("<br><b>MG Type &nbsp;&nbsp;</b><select id=\"temp_pc_mg_type" + "\"></select>");
        $("#tempInput").append("<br><b>MG Levels </b><input type='text' id=\'temp_pc_mg_levels" + "\' maxlength='4'>");
        populateList("mg","","#temp_pc_mg_type");
    }
    else if(pc_type == "gamg") {
        $("#tempInput").append("<br><b>GAMG Type &nbsp;&nbsp;</b><select id=\"temp_pc_gamg_type" + "\"></select>");
        $("#tempInput").append("<br><b>GAMG Levels </b><input type='text' id=\'temp_pc_gamg_levels" + "\' maxlength='4'>");
        populateList("gamg","","#temp_pc_gamg_type");
    }
}

//deletes all children from matInfo
function deleteAllChildren(endtag) {

    var numChildren = getNumChildren(matInfo, endtag);

    for(var i=0; i<numChildren; i++) {
        var childEndtag = endtag + "_" + i;

        if(getNumChildren(matInfo, childEndtag) > 0)//this child has more children
        {
            removeAllChildren(childEndtag);//recursive call to remove all children of that child
        }
        delete matInfo[childEndtag];
        $("#solver" + childEndtag).remove();
    }

    //adjust variables in matInfo (shouldn't really be needed)
    if(matInfo[endtag].pc_type == "mg") {
        matInfo[endtag].pc_mg_levels = 0;
    }
    else if(matInfo[endtag].pc_type == "gamg") {
        matInfo[endtag].pc_gamg_levels = 0;
    }
    else if(matInfo[endtag].pc_type == "fieldsplit") {
        matInfo[endtag].pc_fieldsplit_blocks = 0;
    }
}
