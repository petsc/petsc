//Make an array. Each array element (matInfo[0]. etc) will have all of the information of the questions.
var matInfo = [];
var currentAsk = "0";//start at id=0. then 00 01, then 000 001 010 011 etc if splitting two every time.
var askedA0 = false;//a one-way flag to record if A0 was asked
var finishedAsking = false;//whether input form has finished (when finished, stop pulling default options from sawsInfo?)

//variables used to collect saws information
var sawsInfo = [];
var fieldsplitKeywords = [];//temperature, omega, etc (the index is the number to put after the a-div)

//Use for pcmg
var mgLevelLocation = ""; //where to put the mg level data once the highest level is determined. put in same level as coarse. this location keeps on getting overwritten every time mg_levels_n is encountered


//  This function is run when the page is first visited
//-----------------------------------------------------
$(document).ready(function(){

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

    //create div 'o-1' for displaying SAWs options
    $("#divPc").append("<div id=\"o-1\"> </div>");

    //create a button for user to click after saws data is done loading
    $("#divPc").append("<input type=button value='Please click when options are done loading\nto set defaults for input form' style='margin-left:20px;' id='doneLoading'>");

    // get and display SAWs options
    SAWsGetAndDisplayDirectory("","#variablesInfo");//this #variablesInfo variable only appears here

    addEventHandlers();
});