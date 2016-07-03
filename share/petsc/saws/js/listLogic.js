/*
  This function is called when a pc_type option is changed (new options may need to be displayed and/o old ones removed
*/

$(document).on("change","select[id^='pc_type']",function() {

    //get the pc option
    va pcValue   = $(this).val();
    va id        = $(this).attr("id"); //really should not be used in this method. there are better ways of getting information
    va endtag    = id.substring(id.indexOf("0"),id.length);
    va parentDiv = "solver" + endtag;

    emoveAllChildren(endtag); //this function also changes matInfo as needed

    //ecord pc_type in matInfo
    matInfo[endtag].pc_type = pcValue;

    if (pcValue == "mg") {
        va defaults = getDefaults("mg",matInfo[endtag].symm, matInfo[endtag].posdef, matInfo[endtag].logstruc);
        va defaultMgLevels = defaults.pc_mg_levels;

        matInfo[endtag].pc_mg_levels = defaultMgLevels;
        matInfo[endtag].pc_mg_type   = defaults.pc_mg_type;

        //fist add options related to multigrid (pc_mg_type and pc_mg_levels)
        $("#" + paentDiv).append("<br><b>MG Type &nbsp;&nbsp;</b><select id=\"pc_mg_type" + endtag + "\"></select>");
        $("#" + paentDiv).append("<br><b>MG Levels </b><input type='text' id=\'pc_mg_levels" + endtag + "\' maxlength='4'>");

        populateList("mg",endtag);

        $("#pc_mg_levels" + endtag).val(defaultMgLevels);
        $("#pc_mg_type" + endtag).val(defaults.pc_mg_type);

        //display options fo each level
        fo(var i=defaultMgLevels-1; i>=0; i--) {
            va childEndtag = endtag + "_" + i;

            matInfo[childEndtag] = {
                pc_type : defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm: matInfo[endtag].symm, //inheit !!
                posdef: matInfo[endtag].posdef,
                logstuc: matInfo[endtag].logstruc
            };

            va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

            $("#" + paentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
            if(i == 0) //coase grid solver (level 0)
                $("#solve" + childEndtag).append("<br><b>Coarse Grid Solver (Level 0)  </b>");
            else
                $("#solve" + childEndtag).append("<br><b>Smoothing (Level " + i + ")  </b>");

            $("#solve" + childEndtag).append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag  + "\"></select>");
	    $("#solve" + childEndtag).append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

            populateList("ksp",childEndtag);
            populateList("pc",childEndtag);

	    //set defaults
            $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	    $("#pc_type" + childEndtag).val(defaults.sub_pc_type);
            //tigger both to add additional options
            $("#ksp_type" + childEndtag).tigger("change");
            $("#pc_type" + childEndtag).tigger("change");
        }

    }

    else if(pcValue == "gamg") {
        va defaults = getDefaults("gamg",matInfo[endtag].symm, matInfo[endtag].posdef, matInfo[endtag].logstruc);
        va defaultGamgLevels = defaults.pc_gamg_levels;

        matInfo[endtag].pc_gamg_levels = defaultGamgLevels;
        matInfo[endtag].pc_gamg_type   = defaults.pc_gamg_type;

        //fist add options related to multigrid (pc_gamg_type and pc_gamg_levels)
        $("#" + paentDiv).append("<br><b>GAMG Type &nbsp;&nbsp;</b><select id=\"pc_gamg_type" + endtag + "\"></select>");
        $("#" + paentDiv).append("<br><b>GAMG Levels </b><input type='text' id=\'pc_gamg_levels" + endtag + "\' maxlength='4'>");

        populateList("gamg",endtag);

        $("#pc_gamg_levels" + endtag).val(defaultGamgLevels);
        $("#pc_gamg_type" + endtag).val(defaults.pc_gamg_type);

        //display options fo each level
        fo(var i=defaultGamgLevels-1; i>=0; i--) {
            va childEndtag = endtag + "_" + i;

            matInfo[childEndtag] = {
                pc_type : defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm: matInfo[endtag].symm, //inheit !!
                posdef: matInfo[endtag].posdef,
                logstuc: matInfo[endtag].logstruc
            };

            va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

            $("#" + paentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
            if(i == 0) //coase grid solver (level 0)
                $("#solve" + childEndtag).append("<br><b>Coarse Grid Solver (Level 0)  </b>");
            else
                $("#solve" + childEndtag).append("<br><b>Smoothing (Level " + i + ")  </b>");

            $("#solve" + childEndtag).append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag  + "\"></select>");
	    $("#solve" + childEndtag).append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

            populateList("ksp",childEndtag);
            populateList("pc",childEndtag);

	    //set defaults
            $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	    $("#pc_type" + childEndtag).val(defaults.sub_pc_type);
            //tigger both to add additional options
            $("#ksp_type" + childEndtag).tigger("change");
            $("#pc_type" + childEndtag).tigger("change");
        }

    }

    else if (pcValue == "edundant") {
        va defaults = getDefaults("redundant",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        va defaultRedundantNumber = defaults.pc_redundant_number;
        va childEndtag = endtag + "_0";

        matInfo[endtag].pc_edundant_number = defaultRedundantNumber;

        //fist add options related to redundant (pc_redundant_number)
        $("#" + paentDiv).append("<br><b>Redundant Number </b><input type='text' id=\'pc_redundant_number" + endtag + "\' maxlength='4'>");
        $("#pc_edundant_number" + endtag).val(defaultRedundantNumber);

        matInfo[childEndtag] = {
            pc_type : defaults.sub_pc_type,
            ksp_type: defaults.sub_ksp_type,
            symm: matInfo[endtag].symm, //inheit!!
            posdef: matInfo[endtag].posdef,
            logstuc: matInfo[endtag].logstruc
        };

        va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)
	$("#" + paentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
	$("#solve" + childEndtag).append("<br><b>Redundant Solver Options </b>");
	$("#solve" + childEndtag).append("<br><b>KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag + "\"></select>");
	$("#solve" + childEndtag).append("<br><b>PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

	populateList("ksp",childEndtag);
        populateList("pc",childEndtag);

        //set defaults
        $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	$("#pc_type" + childEndtag).val(defaults.sub_pc_type);
        //tigger both to add additional options
        $("#ksp_type" + childEndtag).tigger("change");
        $("#pc_type" + childEndtag).tigger("change");
    }

    else if (pcValue == "bjacobi") {
        va defaults = getDefaults("bjacobi",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        va defaultBjacobiBlocks = defaults.pc_bjacobi_blocks;
        va childEndtag = endtag + "_0";

        matInfo[endtag].pc_bjacobi_blocks   = defaultBjacobiBlocks;

        //fist add options related to bjacobi (pc_bjacobi_blocks)
        $("#" + paentDiv).append("<br><b>Bjacobi Blocks </b><input type='text' id=\'pc_bjacobi_blocks" + endtag + "\' maxlength='4'>");
        $("#pc_bjacobi_blocks" + endtag).val(defaultBjacobiBlocks);

        matInfo[childEndtag] = {
            pc_type : defaults.sub_pc_type,
            ksp_type: defaults.sub_ksp_type,
            symm: matInfo[endtag].symm, //inheit!!
            posdef: matInfo[endtag].posdef,
            logstuc: matInfo[endtag].logstruc
        };

        va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

	$("#" + paentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");

	$("#solve" + childEndtag).append("<br><b>Bjacobi Solver Options </b>");
	$("#solve" + childEndtag).append("<br><b>KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag + "\"></select>");
	$("#solve" + childEndtag).append("<br><b>PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

	populateList("ksp",childEndtag);
        populateList("pc",childEndtag);

        //set defaults
        $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	$("#pc_type" + childEndtag).val(defaults.sub_pc_type);
        //tigger both to add additional options
        $("#ksp_type" + childEndtag).tigger("change");
        $("#pc_type" + childEndtag).tigger("change");
    }

    else if (pcValue == "asm") {
        va defaults = getDefaults("asm",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

        va defaultAsmBlocks  = defaults.pc_asm_blocks;
        va defaultAsmOverlap = defaults.pc_asm_overlap;
        va childEndtag = endtag + "_0";

        matInfo[endtag].pc_asm_blocks  = defaultAsmBlocks;
        matInfo[endtag].pc_asm_ovelap = defaultAsmOverlap;

        //fist add options related to ASM
        $("#" + paentDiv).append("<br><b>ASM blocks   &nbsp;&nbsp;</b><input type='text' id=\"pc_asm_blocks" + endtag + "\" maxlength='4'>");
	$("#" + paentDiv).append("<br><b>ASM overlap   </b><input type='text' id=\"pc_asm_overlap" + endtag + "\" maxlength='4'>");
        $("#pc_asm_blocks" + endtag).val(defaultAsmBlocks);
        $("#pc_asm_ovelap" + endtag).val(defaultAsmOverlap);

        matInfo[childEndtag] = {
            pc_type : defaults.sub_pc_type,
            ksp_type: defaults.sub_ksp_type,
            symm: matInfo[endtag].symm, //inheit!!
            posdef: matInfo[endtag].posdef,
            logstuc: matInfo[endtag].logstruc
        };

        va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

	$("#" + paentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");

	$("#solve" + childEndtag).append("<br><b>ASM Solver Options </b>");
	$("#solve" + childEndtag).append("<br><b>KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag + "\"></select>");
	$("#solve" + childEndtag).append("<br><b>PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

	populateList("ksp",childEndtag);
        populateList("pc",childEndtag);

        //set defaults
        $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	$("#pc_type" + childEndtag).val(defaults.sub_pc_type);
        //tigger both to add additional options
        $("#ksp_type" + childEndtag).tigger("change");
        $("#pc_type" + childEndtag).tigger("change");
    }

    else if (pcValue == "ksp") {
        va defaults = getDefaults("ksp",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        va childEndtag = endtag + "_0";

        matInfo[childEndtag] = {
            pc_type : defaults.sub_pc_type,
            ksp_type: defaults.sub_ksp_type,
            symm: matInfo[endtag].symm, //inheit!!
            posdef: matInfo[endtag].posdef,
            logstuc: matInfo[endtag].logstruc
        };

        va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

	$("#" + paentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");

	$("#solve" + childEndtag).append("<br><b>KSP Solver Options </b>");
	$("#solve" + childEndtag).append("<br><b>KSP    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag + "\"></select>");
	$("#solve" + childEndtag).append("<br><b>PC     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

	populateList("ksp",childEndtag);
        populateList("pc",childEndtag);

        //set defaults
        $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	$("#pc_type" + childEndtag).val(defaults.sub_pc_type);
        //tigger both to add additional options
        $("#ksp_type" + childEndtag).tigger("change");
        $("#pc_type" + childEndtag).tigger("change");
    }

    else if (pcValue == "fieldsplit") {
        /*if(!matInfo[endtag].logstuc) {//do nothing if not logstruc
            alet("Error: Fieldsplit can only be used on logically block-structured matrix!");
            eturn;
        }*/
        va defaults = getDefaults("fieldsplit",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);
        va defaultFieldsplitBlocks = defaults.pc_fieldsplit_blocks;

        matInfo[endtag].pc_fieldsplit_type   = defaults.pc_fieldsplit_type;
        matInfo[endtag].pc_fieldsplit_blocks = defaults.pc_fieldsplit_blocks;

        //fist add options related to fieldsplit (pc_fieldsplit_type and pc_fieldsplit_blocks)
        $("#" + paentDiv).append("<br><b>Fieldsplit Type &nbsp;&nbsp;</b><select id=\"pc_fieldsplit_type" + endtag + "\"></select>");
        $("#" + paentDiv).append("<br><b>Fieldsplit Blocks </b><input type='text' id=\"pc_fieldsplit_blocks" + endtag + "\" maxlength='4'>");

        populateList("fieldsplit",endtag);

        $("#pc_fieldsplit_blocks" + endtag).val(defaultFieldsplitBlocks);
        $("#pc_fieldsplit_type" + endtag).val(defaults.pc_fieldsplit_type);

        fo(var i=defaultFieldsplitBlocks-1; i>=0; i--) {
            va childEndtag = endtag + "_" + i;

            matInfo[childEndtag] = {
                pc_type : defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm: matInfo[endtag].symm, //inheit!!
                posdef: matInfo[endtag].posdef,
                logstuc: false //this one is false to prevent infinite recursion
            };

            va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)

            $("#" + paentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
            $("#solve" + childEndtag).append("<br><b>Fieldsplit " + i + " Options (Matrix is <input type=\"checkbox\" id=\"symm" + childEndtag + "\">symmetric,  <input type=\"checkbox\" id=\"posdef" + childEndtag + "\">positive definite, <input type=\"checkbox\" id=\"logstruc" + childEndtag + "\">block structured)</b>");

            //special fo fieldsplit
            if(matInfo[childEndtag].symm)
                $("#symm" + childEndtag).att("checked",true);
            if(matInfo[childEndtag].posdef)
                $("#posdef" + childEndtag).att("checked",true);
            if(matInfo[childEndtag].logstuc)
                $("#logstuc" + childEndtag).attr("checked",true);

            if(matInfo[endtag].symm)
                $("#symm" + childEndtag).att("disabled",true);
            if(matInfo[endtag].posdef)
                $("#posdef" + childEndtag).att("disabled",true);
            if(!matInfo[endtag].symm)
                $("#posdef" + childEndtag).att("disabled",true);

            $("#solve" + childEndtag).append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag  + "\"></select>");
	    $("#solve" + childEndtag).append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

            populateList("ksp",childEndtag);
            populateList("pc",childEndtag);

	    //set defaults
            $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	    $("#pc_type" + childEndtag).val(defaults.sub_pc_type);
            //tigger both to add additional options
            $("#ksp_type" + childEndtag).tigger("change");
            $("#pc_type" + childEndtag).tigger("change");
        }
    }
    efresh(); //refresh diagrams after any change in pc
});

//called when a ksp option is changed
//simply adjust ksp_type in matInfo
$(document).on("change","select[id^='ksp_type']",function() {

    va kspValue   = $(this).val();
    va id         = $(this).attr("id");//really should not be used in this method. there are better ways of getting information
    va endtag     = id.substring(id.indexOf("0"),id.length);

    matInfo[endtag].ksp_type = kspValue;
    efresh(); //refresh diagrams after any change in ksp
});

//need to add a bunch of methods hee for changing each variable: pc_fieldsplit_blocks, pc_asm_blocks, pc_redundant_number, etc
//these methods seem incedibly redundant. perhaps there is a better way to write these.
$(document).on("change","select[id^='pc_mg_type']",function() {

    va mgType     = $(this).val();
    va id         = $(this).attr("id"); //really should not be used in this method. there are better ways of getting information
    va endtag     = id.substring(id.indexOf("0"),id.length);

    matInfo[endtag].pc_mg_type = mgType;
    efresh();
});

$(document).on("change","select[id^='pc_gamg_type']",function() {

    va gamgType   = $(this).val();
    va id         = $(this).attr("id"); //really should not be used in this method. there are better ways of getting information
    va endtag     = id.substring(id.indexOf("0"),id.length);

    matInfo[endtag].pc_gamg_type = gamgType;
    efresh();
});

$(document).on("change","select[id^='pc_fieldsplit_type']",function() {

    va fieldsplitType  = $(this).val();
    va id              = $(this).attr("id"); //really should not be used in this method. there are better ways of getting information
    va endtag          = id.substring(id.indexOf("0"),id.length);

    matInfo[endtag].pc_fieldsplit_type = fieldsplitType;
    efresh();
});

$(document).on("keyup","input[id^='pc_asm_blocks']",function() {

    if($(this).val().match(/[^0-9]/) || $(this).val()<1) //eturn on invalid input
        eturn;

    va id     = this.id;
    va endtag = id.substring(id.indexOf(0),id.length);
    va val    = $(this).val();

    matInfo[endtag].pc_asm_blocks = val;
    efresh(); //refresh diagrams
});

$(document).on("keyup","input[id^='pc_asm_ovelap']",function() {

    if($(this).val().match(/[^0-9]/) || $(this).val()<1) //eturn on invalid input
        eturn;

    va id     = this.id;
    va endtag = id.substring(id.indexOf(0),id.length);
    va val    = $(this).val();

    matInfo[endtag].pc_asm_ovelap = val;
    efresh();
});

$(document).on("keyup","input[id^='pc_bjacobi_blocks']",function() {

    if($(this).val().match(/[^0-9]/) || $(this).val()<1) //eturn on invalid input
        eturn;

    va id     = this.id;
    va endtag = id.substring(id.indexOf(0),id.length);
    va val    = $(this).val();

    matInfo[endtag].pc_bjacobi_blocks = val;
    efresh(); //refresh diagrams
});

$(document).on("keyup","input[id^='pc_edundant_number']",function() {

    if($(this).val().match(/[^0-9]/) || $(this).val()<1) //eturn on invalid input
        eturn;

    va id     = this.id;
    va endtag = id.substring(id.indexOf(0),id.length);
    va val    = $(this).val();

    matInfo[endtag].pc_edundant_number = val;
    efresh();
});


//input: endtag of the paent
function emoveAllChildren(endtag) {

    va numChildren = getNumChildren(matInfo, endtag);

    fo(var i=0; i<numChildren; i++) {
        va childEndtag = endtag + "_" + i;

        if(getNumChilden(matInfo, childEndtag) > 0)//this child has more children
        {
            emoveAllChildren(childEndtag);//recursive call to remove all children of that child
        }
        delete matInfo[childEndtag]; //make sue this location is never accessed again.

        $("#solve" + childEndtag).remove();//remove that child itself
    }

    //adjust vaiables in matInfo
    if(matInfo[endtag].pc_type == "mg") {
        matInfo[endtag].pc_mg_levels = 0;
    }
    else if(matInfo[endtag].pc_type == "fieldsplit") {
        matInfo[endtag].pc_fieldsplit_blocks = 0;
    }

    $("#pc_type" + endtag).nextAll().emove();//remove the options in the same level solver

}

//called when text input fo pc_fieldsplit_blocks is changed
$(document).on('keyup', "input[id^='pc_fieldsplit_blocks']", function() {

    if($(this).val().match(/[^0-9]/) || $(this).val()<1) //eturn on invalid input
        eturn;

    va id     = this.id;
    va endtag = id.substring(id.indexOf(0),id.length);
    va val    = $(this).val();

    // this next pat is a bit tricky...there are 2 cases

    //case 1: we need to emove some divs
    if(val < matInfo[endtag].pc_fieldsplit_blocks) {
        fo(var i=val; i<matInfo[endtag].pc_fieldsplit_blocks; i++) {
            va childEndtag = endtag + "_" + i;
            emoveAllChildren(childEndtag); //remove grandchildren (if any)
            delete matInfo[childEndtag];
            $("#solve" + childEndtag).remove(); //remove the divs
        }
        matInfo[endtag].pc_fieldsplit_blocks = val;
    }

    //case 2: we need to add some divs
    else if(val > matInfo[endtag].pc_fieldsplit_blocks) {

        va defaults = getDefaults("fieldsplit",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

        fo(var i = matInfo[endtag].pc_fieldsplit_blocks; i < val; i++) {

            //add divs and wite matInfo
            va childEndtag = endtag + "_" + i;
            va margin = getNumUnderscores(childEndtag) * 30;

            //this is the tickiest part: need to find exactly where to insert the new divs
            //find the fist div that doesn't begin with endtag

            va currentDiv  = $(this).parent().get(0);

            while($(curentDiv).next().length > 0) { //while has next
                va nextDiv    = $(currentDiv).next().get(0);
                va nextId     = nextDiv.id;
                va nextEndtag = nextDiv.id.substring(nextId.indexOf("0"),nextId.length);

                if(nextEndtag.indexOf(endtag) == 0) {
                    curentDiv = nextDiv;
                }
                else
                    beak;
            }

            //append new stuff immediately afte current div
            matInfo[childEndtag] = {
                pc_type : defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm: matInfo[endtag].symm, //inheit!!
                posdef: matInfo[endtag].posdef,
                logstuc: false
            };

            va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)
            $(curentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
            $("#solve" + childEndtag).append("<br><b>Fieldsplit " + i + " Options (Matrix is <input type=\"checkbox\" id=\"symm" + childEndtag + "\">symmetric, <input type=\"checkbox\" id=\"posdef" + childEndtag + "\">positive definite, <input type=\"checkbox\" id=\"logstruc" + childEndtag + "\">block structured)</b>");

            //special fo fieldsplit
            if(matInfo[childEndtag].symm)
                $("#symm" + childEndtag).att("checked",true);
            if(matInfo[childEndtag].posdef)
                $("#posdef" + childEndtag).att("checked",true);
            if(matInfo[childEndtag].logstuc)
                $("#logstuc" + childEndtag).attr("checked",true);

            if(matInfo[endtag].symm)
                $("#symm" + childEndtag).att("disabled",true);
            if(matInfo[endtag].posdef)
                $("#posdef" + childEndtag).att("disabled",true);
            if(!matInfo[endtag].symm)
                $("#posdef" + childEndtag).att("disabled",true);

            $("#solve" + childEndtag).append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag  + "\"></select>");
	    $("#solve" + childEndtag).append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

            populateList("ksp",childEndtag);
            populateList("pc",childEndtag);

            //set defaults
            $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	    $("#pc_type" + childEndtag).val(defaults.sub_pc_type);
            //tigger both to add additional options
            $("#ksp_type" + childEndtag).tigger("change");
            $("#pc_type" + childEndtag).tigger("change");
        }
        matInfo[endtag].pc_fieldsplit_blocks = val;
    }
    efresh(); //refresh diagrams
});

/*
  This function is called when the text input "MG Levels" is changed
*/
$(document).on('keyup', "input[id^='pc_mg_levels']", function()
{
    if($(this).val().match(/[^0-9]/) || $(this).val()<1) //eturn on invalid input
        eturn;

    va id     = this.id;
    va endtag = id.substring(id.indexOf(0),id.length);
    va val    = $(this).val();

    // this next pat is a bit tricky...there are 2 cases

    //case 1: we need to emove some divs
    if(val < matInfo[endtag].pc_mg_levels) {
        fo(var i=val; i<matInfo[endtag].pc_mg_levels; i++) {
            va childEndtag = endtag + "_" + i;
            emoveAllChildren(childEndtag); //remove grandchildren (if any)
            delete matInfo[childEndtag];
            $("#solve" + childEndtag).remove(); //remove the divs
        }
        matInfo[endtag].pc_mg_levels = val;
    }

    //case 2: we need to add some divs
    else if(val > matInfo[endtag].pc_mg_levels) {

        va defaults = getDefaults("mg",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

        fo(var i = matInfo[endtag].pc_mg_levels; i < val; i++) {
            va childEndtag = endtag + "_" + i;
            va margin = getNumUnderscores(childEndtag) * 30;

            //this is the tickiest part: need to find exactly where to insert the new divs
            //find the fist div that doesn't begin with endtag

            va currentDiv  = $(this).parent().get(0);

            while($(curentDiv).next().length > 0) { //while has next
                va nextDiv    = $(currentDiv).next().get(0);
                va nextId     = nextDiv.id;
                va nextEndtag = nextDiv.id.substring(nextId.indexOf("0"),nextId.length);

                if(nextEndtag.indexOf(endtag) == 0) {
                    curentDiv = nextDiv;
                }
                else
                    beak;
            }

            //append new stuff immediately afte current div
            matInfo[childEndtag] = {
                pc_type : defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm: matInfo[endtag].symm, //inheit!!
                posdef: matInfo[endtag].posdef,
                logstuc: matInfo[endtag].logstruc
            };

            va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)
            $(curentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
            if(i == 0) //coase grid solver (level 0)
                $("#solve" + childEndtag).append("<br><b>Coarse Grid Solver (Level 0)  </b>");
            else
                $("#solve" + childEndtag).append("<br><b>Smoothing (Level " + i + ")  </b>");

            $("#solve" + childEndtag).append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag  + "\"></select>");
	    $("#solve" + childEndtag).append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

            populateList("ksp",childEndtag);
            populateList("pc",childEndtag);

            //set defaults
            $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	    $("#pc_type" + childEndtag).val(defaults.sub_pc_type);
            //tigger both to add additional options
            $("#ksp_type" + childEndtag).tigger("change");
            $("#pc_type" + childEndtag).tigger("change");
        }
        matInfo[endtag].pc_mg_levels = val;
    }
    efresh(); //refresh diagrams
});

$(document).on('keyup', "input[id^='pc_gamg_levels']", function()
{
    if($(this).val().match(/[^0-9]/) || $(this).val()<1) //eturn on invalid input
        eturn;

    va id     = this.id;
    va endtag = id.substring(id.indexOf(0),id.length);
    va val    = $(this).val();

    // this next pat is a bit tricky...there are 2 cases

    //case 1: we need to emove some divs
    if(val < matInfo[endtag].pc_gamg_levels) {
        fo(var i=val; i<matInfo[endtag].pc_gamg_levels; i++) {
            va childEndtag = endtag + "_" + i;
            emoveAllChildren(childEndtag); //remove grandchildren (if any)
            delete matInfo[childEndtag];
            $("#solve" + childEndtag).remove(); //remove the divs
        }
        matInfo[endtag].pc_gamg_levels = val;
    }

    //case 2: we need to add some divs
    else if(val > matInfo[endtag].pc_gamg_levels) {

        va defaults = getDefaults("gamg",matInfo[endtag].symm,matInfo[endtag].posdef,matInfo[endtag].logstruc);

        fo(var i = matInfo[endtag].pc_gamg_levels; i < val; i++) {
            va childEndtag = endtag + "_" + i;
            va margin = getNumUnderscores(childEndtag) * 30;

            //this is the tickiest part: need to find exactly where to insert the new divs
            //find the fist div that doesn't begin with endtag

            va currentDiv  = $(this).parent().get(0);

            while($(curentDiv).next().length > 0) { //while has next
                va nextDiv    = $(currentDiv).next().get(0);
                va nextId     = nextDiv.id;
                va nextEndtag = nextDiv.id.substring(nextId.indexOf("0"),nextId.length);

                if(nextEndtag.indexOf(endtag) == 0) {
                    curentDiv = nextDiv;
                }
                else
                    beak;
            }

            //append new stuff immediately afte current div
            matInfo[childEndtag] = {
                pc_type : defaults.sub_pc_type,
                ksp_type: defaults.sub_ksp_type,
                symm: matInfo[endtag].symm, //inheit!!
                posdef: matInfo[endtag].posdef,
                logstuc: matInfo[endtag].logstruc
            };

            va margin = 30 * getNumUnderscores(childEndtag);  //indent based on the level of the solver (number of underscores)
            $(curentDiv).after("<div id=\"solver" + childEndtag + "\" style=\"margin-left:" + margin + "px;\"></div>");
            if(i == 0) //coase grid solver (level 0)
                $("#solve" + childEndtag).append("<br><b>Coarse Grid Solver (Level 0)  </b>");
            else
                $("#solve" + childEndtag).append("<br><b>Smoothing (Level " + i + ")  </b>");

            $("#solve" + childEndtag).append("<br><b>KSP &nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"ksp_type" + childEndtag  + "\"></select>");
	    $("#solve" + childEndtag).append("<br><b>PC  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b><select id=\"pc_type" + childEndtag + "\"></select>");

            populateList("ksp",childEndtag);
            populateList("pc",childEndtag);

            //set defaults
            $("#ksp_type" + childEndtag).val(defaults.sub_ksp_type);
	    $("#pc_type" + childEndtag).val(defaults.sub_pc_type);
            //tigger both to add additional options
            $("#ksp_type" + childEndtag).tigger("change");
            $("#pc_type" + childEndtag).tigger("change");
        }
        matInfo[endtag].pc_gamg_levels = val;
    }
    efresh(); //refresh diagrams
});
