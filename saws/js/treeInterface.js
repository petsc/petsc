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
    var index = getIndex(matInfo,endtag);

    var oldPc = matInfo[index].pc_type;
    var newPc = $("#temp_pc_type").val();

    var changedPc = true;
    if(oldPc == newPc)
        changedPc = false;

    matInfo[index].pc_type = $("#temp_pc_type").val();
    matInfo[index].ksp_type = $("#temp_ksp_type").val();

    var pc_type = matInfo[index].pc_type;

    if(pc_type == "fieldsplit") { //extra options for fieldsplit
        //first make sure inputted options are valid
        var blocks = $("#temp_pc_fieldsplit_blocks").val();
        if(blocks.match(/[^0-9]/) || blocks < 1) {
            alert("Error: Must have at least 1 block for fieldsplit");
            return 0;
        }

        matInfo[index].pc_fieldsplit_type = $("#temp_pc_fieldsplit_type").val();

        if(matInfo[index].pc_fieldsplit_blocks == undefined)
            matInfo[index].pc_fieldsplit_blocks = 0;

        //case 1: more blocks required
        if(matInfo[index].pc_fieldsplit_blocks  < blocks) {
            matInfo[index].pc_fieldsplit_blocks = $("#temp_pc_fieldsplit_blocks").val();
            generateChildren(endtag); //simply ask for the children to be generated
        }
        //case 2: some blocks need to be removed
        else if(matInfo[index].pc_fieldsplit_blocks > blocks) {
            for(var i=blocks; i<matInfo[index].pc_fieldsplit_blocks; i++) {
                var childEndtag = endtag + "_" + i;
                deleteAllChildren(childEndtag); //delete the necessary children
            }
            matInfo[index].pc_fieldsplit_blocks = $("#temp_pc_fieldsplit_blocks").val();
        }

        return 1;
    }
    else if(pc_type == "mg") { //extra options for mg
        matInfo[index].pc_mg_type = $("#temp_pc_mg_type").val();
        matInfo[index].pc_mg_levels = $("#temp_pc_mg_levels").val();
    }
    else if(pc_type == "gamg") {
        matInfo[index].pc_gamg_type = $("#temp_pc_gamg_type").val();
        matInfo[index].pc_gamg_levels = $("#temp_pc_gamg_levels").val();
    }
    else if(pc_type == "bjacobi") {
        matInfo[index].pc_bjacobi_blocks = $("#temp_pc_bjacobi_blocks").val();
    }
    else if(pc_type == "redundant") {
        matInfo[index].pc_redundant_number = $("#temp_pc_redundant_number").val();
    }
    else if(pc_type == "asm") {
        matInfo[index].pc_asm_blocks = $("#temp_pc_asm_blocks").val();
        matInfo[index].pc_asm_overlap = $("#temp_pc_asm_overlap").val();
    }
    else if(pc_type == "ksp") {
        //ksp doesn't have any additional options, but it has one child
    }

}

//this function ensures that all children are properly generated and initialized
function generateChildren(endtag) {

    var index = getIndex(matInfo,endtag);
    var pc_type = matInfo[index].pc_type;

    if(pc_type == "bjacobi") { //needs 1 child
        //check if that 1 child already exists. if so, then stop. otherwise, generate  that child, select the defaults, and recursively ensure that the new child also has the appropriate children
        var childEndtag = endtag + "_0";
        var childIndex = getIndex(matInfo,childEndtag);
        if(childIndex == -1) { //generate child
            var defaults = getDefaults("bjacobi",matInfo[index].symm,matInfo[index].posdef,matInfo[index].logstruc);

            var writeLoc = matInfo.length;
            matInfo[writeLoc] = {
                endtag:   childEndtag,
                pc_type:  defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm:     matInfo[index].symm, //inherit !!
                posdef:   matInfo[index].posdef,
                logstruc: matInfo[index].logstruc
            }
            generateChildren(childEndtag);
        }
    }
    else if(pc_type == "fieldsplit") {
        for(var i=0; i<matInfo[index].pc_fieldsplit_blocks; i++) {
            var childEndtag = endtag + "_" + i;
            var childIndex = getIndex(matInfo,childEndtag);
            if(childIndex == -1) { //generate child
                var defaults = getDefaults("fieldsplit",matInfo[index].symm,matInfo[index].posdef,matInfo[index].logstruc);

                var writeLoc = matInfo.length;
                matInfo[writeLoc] = {
                    endtag:   childEndtag,
                    pc_type:  defaults.sub_pc_type,
                    ksp_type: defaults.sub_ksp_type,
                    symm:     matInfo[index].symm, //inherit !!
                    posdef:   matInfo[index].posdef,
                    logstruc: false //to prevent infinite recursion
                }
                generateChildren(childEndtag);
            }
        }
    }
    else if(pc_type == "mg") {
        for(var i=0; i<matInfo[index].pc_mg_levels; i++) {
            var childEndtag = endtag + "_" + i;
            var childIndex = getIndex(matInfo,childEndtag);
            if(childIndex == -1) { //generate child
                var defaults = getDefaults("mg",matInfo[index].symm,matInfo[index].posdef,matInfo[index].logstruc);

                var writeLoc = matInfo.length;
                matInfo[writeLoc] = {
                    endtag:   childEndtag,
                    pc_type:  defaults.sub_pc_type,
                    ksp_type: defaults.sub_ksp_type,
                    symm:     matInfo[index].symm, //inherit !!
                    posdef:   matInfo[index].posdef,
                    logstruc: true
                }
                generateChildren(childEndtag);
            }
        }
    }
    else if(pc_type == "gamg") {
        for(var i=0; i<matInfo[index].pc_gamg_levels; i++) {
            var childEndtag = endtag + "_" + i;
            var childIndex = getIndex(matInfo,childEndtag);
            if(childIndex == -1) { //generate child
                var defaults = getDefaults("gamg",matInfo[index].symm,matInfo[index].posdef,matInfo[index].logstruc);

                var writeLoc = matInfo.length;
                matInfo[writeLoc] = {
                    endtag:   childEndtag,
                    pc_type:  defaults.sub_pc_type,
                    ksp_type: defaults.sub_ksp_type,
                    symm:     matInfo[index].symm, //inherit !!
                    posdef:   matInfo[index].posdef,
                    logstruc: true
                }
                generateChildren(childEndtag);
            }
        }
    }
    else if(pc_type == "ksp") {
        var childEndtag = endtag + "_0";
        var childIndex = getIndex(matInfo,childEndtag);
        if(childIndex == -1) { //generate child
            var defaults = getDefaults("ksp",matInfo[index].symm,matInfo[index].posdef,matInfo[index].logstruc);

            var writeLoc = matInfo.length;
            matInfo[writeLoc] = {
                endtag:   childEndtag,
                pc_type:  defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm:     matInfo[index].symm, //inherit !!
                posdef:   matInfo[index].posdef,
                logstruc: matInfo[index].logstruc
            }
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
    else
        alert("Error in fields");

});

//this function displays the appropriate options for each pc_type when pc_type dropdown is changed (for example, bjacobi blocks, fieldsplit type, etc)
$(document).on("change","select[id='temp_pc_type']", function(){

    var endtag = boxEndtag;
    var index  = getIndex(matInfo,endtag);
    var pc_type = $(this).val();

    //first, remove all the existing suboptions
    $("#temp_pc_type").nextAll().remove();//remove the options in the same level solver

    if(pc_type == "fieldsplit") { //add the extra options that fieldsplit requires
        $("#tempInput").append("<br><b>Fieldsplit Type &nbsp;&nbsp;</b><select id=\"temp_pc_fieldsplit_type" + "\"></select>");
        $("#tempInput").append("<br><b>Fieldsplit Blocks </b><input type='text' id=\"temp_pc_fieldsplit_blocks" + "\" maxlength='4'>");
        populateFieldsplitList(endtag, "#temp_pc_fieldsplit_type");

        var defaults = "";
    }
    else if(pc_type == "mg") {

    }
    else if(pc_type == "bjacobi") {

    }
    else if(pc_type == "gamg") {

    }
    else if(pc_type == "redundant") {

    }
    else if(pc_type == "asm") {

    }

    $("#tempInput").append("<br><input type=\"button\" value=\"Set Options\" id=\"setOptions\">"); //put the button back because we still want that (was removed in the nextAll().remove())
});

//this shouldn't be in boxTree. this should be in events.
//upon a user click, we present the user with the currently selected options that the user can change
//this method does NOT handle changes in the selected pc option
$(document).on("click","circle[id^='node']",function(){

    var id     = $(this).attr("id");//really should not be used in this method. there are better ways of getting information
    var endtag = id.substring(id.indexOf("0"),id.length);
    var index  = getIndex(matInfo,endtag);
    var parentEndtag = getParent(endtag);
    var parentIndex  = getIndex(matInfo,parentEndtag);
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

    var childNum = endtag.substring(endtag.lastIndexOf("_")+1, endtag.length);

    var solverText = "";
    if(endtag == "0")
        solverText = "<b>Root Solver Options (Matrix is <input type=\"checkbox\" id=\"temp_symm" + "\">symmetric,  <input type=\"checkbox\" id=\"temp_posdef" + "\">positive definite, <input type=\"checkbox\" id=\"temp_logstruc" + "\">block structured)</b>";
    else if(matInfo[parentIndex].pc_type == "bjacobi")
        solverText = "<b>" + "Bjacobi Solver Options" + "</b>";
    else if(matInfo[parentIndex].pc_type == "fieldsplit")
        solverText = "<b>Fieldsplit " + childNum + " Options (Matrix is <input type=\"checkbox\" id=\"temp_symm" + "\">symmetric,  <input type=\"checkbox\" id=\"temp_posdef" + "\">positive definite, <input type=\"checkbox\" id=\"temp_logstruc" + "\">block structured)</b>";
    else if(matInfo[parentIndex].pc_type == "redundant")
        solverText = "<b>" + "Redundant Solver Options" + "</b>";
    else if(matInfo[parentIndex].pc_type == "asm")
        solverText = "<b>" + "ASM Solver Options" + "</b>";
    else if(matInfo[parentIndex].pc_type == "ksp")
        solverText = "<b>" + "KSP Solver Options" + "</b>";
    else if(matInfo[parentIndex].pc_type == "mg" || matInfo[parentIndex].pc_type == "gamg") {
        if(childNum == 0) //coarse grid solver (level 0)
            solverText = "<b> Coarse Grid Solver (Level 0)  </b>";
        else
            solverText = "<b>Smoothing (Level " + childNum + ")  </b>";
    }


    $("#tempInput").append(solverText);
    $("#tempInput").append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"temp_ksp_type" + "\"></select>");
    $("#tempInput").append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"temp_pc_type" + "\"></select>");

    populatePcList(endtag,"#temp_pc_type"); //this is kind of stupid right now. I'll fix this later
    populateKspList(endtag,"#temp_ksp_type");
    $("#temp_pc_type").val(matInfo[index].pc_type);
    $("#temp_ksp_type").val(matInfo[index].ksp_type);


    if(matInfo[index].pc_type == "fieldsplit") { //append extra options for fieldsplit
        $("#tempInput").append("<br><b>Fieldsplit Type &nbsp;&nbsp;</b><select id=\"temp_pc_fieldsplit_type" + "\"></select>");
        $("#tempInput").append("<br><b>Fieldsplit Blocks </b><input type='text' id=\"temp_pc_fieldsplit_blocks" + "\" maxlength='4'>");

        populateFieldsplitList(endtag, "#temp_pc_fieldsplit_type");
        $("#temp_pc_fieldsplit_type").val(matInfo[index].pc_fieldsplit_type);
        $("#temp_pc_fieldsplit_blocks").val(matInfo[index].pc_fieldsplit_blocks);
    }
    else if(matInfo[index].pc_type == "bjacobi") { //append extra options for bjacobi
        $("#tempInput").append("<br><b>Bjacobi Blocks </b><input type='text' id=\'temp_pc_bjacobi_blocks" + "\' maxlength='4'>");
        $("#temp_pc_bjacobi_blocks").val(matInfo[index].pc_bjacobi_blocks);
    }
    else if(matInfo[index].pc_type == "redundant") {
        $("#tempInput").append("<br><b>Redundant Number </b><input type='text' id=\'temp_pc_redundant_number" + "\' maxlength='4'>");
        $("#temp_pc_redundant_number").val(matInfo[index].pc_redundant_number);
    }
    else if(matInfo[index].pc_type == "asm") {
        $("#tempInput").append("<br><b>ASM blocks   &nbsp;&nbsp;</b><input type='text' id=\"temp_pc_asm_blocks" + "\" maxlength='4'>");
	$("#tempInput").append("<br><b>ASM overlap   </b><input type='text' id=\"temp_pc_asm_overlap" + "\" maxlength='4'>");
        $("#temp_pc_asm_blocks").val(matInfo[index].pc_asm_blocks);
        $("#temp_pc_asm_overlap").val(matInfo[index].pc_asm_overlap);
    }
    else if(matInfo[index].pc_type == "mg") {
        $("#tempInput").append("<br><b>MG Type &nbsp;&nbsp;</b><select id=\"temp_pc_mg_type" + "\"></select>");
        $("#tempInput").append("<br><b>MG Levels </b><input type='text' id=\'temp_pc_mg_levels" + "\' maxlength='4'>");
        populateMgList(endtag,"#temp_pc_mg_type");
        $("#temp_pc_mg_type").val(matInfo[index].pc_mg_type);
        $("#temp_pc_mg_levels").val(matInfo[index].pc_mg_levels);
    }
    else if(matInfo[index].pc_type == "gamg") {
        $("#tempInput").append("<br><b>GAMG Type &nbsp;&nbsp;</b><select id=\"temp_pc_gamg_type" + "\"></select>");
        $("#tempInput").append("<br><b>GAMG Levels </b><input type='text' id=\'temp_pc_gamg_levels" + "\' maxlength='4'>");
        populateGamgList(endtag,"#temp_pc_gamg_type");
        $("#temp_pc_gamg_type").val(matInfo[index].pc_gamg_type);
        $("#temp_pc_gamg_levels").val(matInfo[index].pc_gamg_levels);
    }

    //append the submit button
    $("#tempInput").append("<br><input type=\"button\" value=\"Set Options\" id=\"setOptions\">");

});

//deletes all children from matInfo
function deleteAllChildren(endtag) {

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
    }

    //adjust variables in matInfo (shouldn't really be needed)
    if(matInfo[index].pc_type == "mg") {
        matInfo[index].pc_mg_levels = 0;
    }
    else if(matInfo[index].pc_type == "fieldsplit") {
        matInfo[index].pc_fieldsplit_blocks = 0;
    }
}
