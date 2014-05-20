function addEventHandlers() {

    //When "Continue" button is clicked ...
    //----------------------------------------
    $("#continueButton").click(function(){

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
        $("#A" + currentAsk).append("<br><b id='matrixText"+currentAsk+"'>A" + currentAsk + " (Symm:"+matInfo[writeLoc].symm+" Posdef:"+matInfo[writeLoc].posdef+" Logstruc:"+matInfo[writeLoc].logstruc +")</b>");
	$("#A" + currentAsk).append("<br><b>KSP &nbsp;</b><select class=\"kspLists\" id=\"kspList" + currentAsk +"\"></select>");
	$("#A" + currentAsk).append("<br><b>PC &nbsp; &nbsp;</b><select class=\"pcLists\" id=\"pcList" + currentAsk +"\"></select>");

        if(matInfo[writeLoc].logstruc) {//if fieldsplit, need to add the fieldsplit type and fieldsplit blocks
            var newDiv = generateDivName("",currentAsk,"fieldsplit");//this div contains the two fieldsplit dropdown menus. as long as first param doesn't contain "_", it will generate assuming it is directly under an A-div which it is
	    var endtag = newDiv.substring(newDiv.lastIndexOf('_'), newDiv.length);
	    $("#A"+currentAsk).append("<div id=\""+newDiv+"\" style='margin-left:"+30+"px;'></div>");//append to the A-div that we just added to the table
            var myendtag = endtag+"0";
	    $("#"+newDiv).append("<b>Fieldsplit Type &nbsp;&nbsp;</b><select class=\"fieldsplitList\" id=\"fieldsplitList" + currentAsk +myendtag+"\"></select>");
            $("#"+newDiv).append("<br><b>Fieldsplit Blocks </b><input type='text' id='fieldsplitBlocks"+currentAsk+myendtag+"\' value='2' maxlength='2' class='fieldsplitBlocks'>");//note that class is fieldsplitBlocks NOT fieldsplitBlocksInput
            populateFieldsplitList("fieldsplitList"+currentAsk+myendtag);
        }

	//populate the kspList and pclist with default options
        if (getSawsIndex(currentAsk) != -1) { //use SAWs options if they exist for this matrix
            var sawsIndex=getSawsIndex(currentAsk);

            var SAWs_kspVal = sawsInfo[sawsIndex].data[getSawsDataIndex(sawsIndex,"")].ksp;//want the ksp where endtag=""
            //SAWs_alternatives ???
            populateKspList("kspList"+currentAsk,null,SAWs_kspVal);

            var SAWs_pcVal = sawsInfo[sawsIndex].data[getSawsDataIndex(sawsIndex,"")].pc;//want the pc where endtag=""
            //SAWs_alternatives ???
	    populatePcList("pcList"+currentAsk,null,SAWs_pcVal);
        } else {//else, use default values
            populateKspList("kspList"+currentAsk,null,"null");
            populatePcList("pcList"+currentAsk,null,"null");
        }

        //manually trigger pclist once because additional options, e.g., detailed info may need to be added
        if($("#pcList"+currentAsk).val()!="fieldsplit")//but DON'T trigger change on fieldsplit because that would append the required A divs twice
	    $("#pcList"+currentAsk).trigger("change");

        currentAsk = matTreeGetNextNode(currentAsk);

        formSet(currentAsk); //reset the form

    });


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
        //alert('when this is called?'); ???
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

    //When "Cmd Options" button is clicked ...
    //----------------------------------------
    var treeDetailed;
    $("#cmdOptionsButton").click(function(){
	$("#treeContainer").html("<div id='tree'> </div>");

	//get the options from the drop-down lists
        solverGetOptions(matInfo);

	//get the number of levels for the tree for scaling purposes
        var matLevelForTree=0;
        for(var i=0; i<matInfo.length; i++)
            if(matInfo[i].id!="-1" && matInfo[i].level>matLevelForTree)
                matLevelForTree=matInfo[i];
        matLevelForTree++;//appears to be 1 greater than the max

	//build the tree
        treeDetailed = false;//tree.js uses this variable to know what information to display
	buildTree(matInfo,matLevelForTree,treeDetailed);

        //show cmdOptions to the screen
        for (var i=0; i<matInfo.length; i++) {
	    if (matInfo[i].id=="-1")//possible junk value created by deletion of adiv
		continue;
	    $("#oCmdOptions" + matInfo[i].id).empty();
            $("#oCmdOptions" + matInfo[i].id).append("<br><br>" + matInfo[i].string);
        }
    });

    $("#solverTreeButtonDetailed").click(function(){
	$("#treeContainer").html("<div id='tree'> </div>");

	//get the options from the drop-down lists
        solverGetOptions(matInfo);

	//get the number of levels for the tree for scaling purposes
        var matLevelForTree=0;
        for(var i=0; i<matInfo.length; i++)
            if(matInfo[i].id!="-1" && matInfo[i].level>matLevelForTree)
                matLevelForTree=matInfo[i];
        matLevelForTree++;//appears to be 1 greater than the max

	//build the tree
        treeDetailed = true;//tree.js uses this variable
	buildTree(matInfo,matLevelForTree,treeDetailed);

        //show cmdOptions to the screen
        for (var i=0; i<matInfo.length; i++) {
	    if (matInfo[i].id=="-1")//possible junk value created by deletion of adiv
		continue;
	    $("#oCmdOptions" + matInfo[i].id).empty();
            $("#oCmdOptions" + matInfo[i].id).append("<br><br>" + matInfo[i].string);
        }
    });

    $("#clearOutput").click(function(){
	for(var i=0; i<matInfo.length; i++)
	    $("#oCmdOptions"+matInfo[i].id).empty();//if matInfo has deleted A-divs, its still okay because id will be "-1" and nothing will be changed
    });

    $("#clearTree").click(function(){
        $("#tree").remove();
    });

}