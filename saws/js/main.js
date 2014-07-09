//Make an array. Each array element (matInfo[0]. etc) will have all of the information of the questions.
var matInfo = [];
//var currentAsk = "0";//start at endtag=0. then 0_0 0_1, then 0_0_0 0_0_1 0_1_0 0_1_1 etc...
//var askedA0 = false;//a one-way flag to record if A0 was asked

//  This function is run when the page is first visited
//-----------------------------------------------------
$(document).ready(function(){

    //reset the form
    //formSet(currentAsk);

    matInfo[0] = {
        endtag: "0",
        logstruc: false,
        symm: false,
        posdef: false,
    }

    //to start, append the first div (div0) in the table and the first pc/ksp options dropdown
    $("#results").append("<div id=\"leftPanel\" style=\"float:left;\"> </div> <div id=\"rightPanel\" style=\"float:left;padding-left:30px;\"></div>");
    $("#leftPanel").append("<div id=\"solver0\"> </div>");

    $("#solver0").append("<b>Root Solver Options (Mat Properties: Symm:<input type=\"checkbox\" id=\"symm0\"> Posdef:<input type=\"checkbox\" id=\"posdef0\"> Logstruc:<input type=\"checkbox\" id=\"logstruc0\">)</b><br>");//text: Solver Level: 0
    $("#solver0").append("<br><b>KSP &nbsp;</b><select id=\"ksp_type0\"></select>");
    $("#solver0").append("<br><b>PC &nbsp;&nbsp;&nbsp;</b><select id=\"pc_type0\"></select>");

    populatePcList("0");
    populateKspList("0");

    $("#pc_type0").trigger("change");//display options for sub-solvers (if any)
    $("#ksp_type0").trigger("change");//just to record ksp. (ask Dr. Smith or Dr. Zhang for proper defaults)

    //display matrix pic. manually add square braces the first time
    //$("#matrixPic").html("<center>" + "\\(\\left[" + getMatrixTex("0") + "\\right]\\)" + "</center>");
    //MathJax.Hub.Queue(["Typeset",MathJax.Hub]);

    addEventHandlers();
});