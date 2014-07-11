//this data structure is used to hold all of the solver options and matrix properties
var matInfo = [];

//  This function is run when the page is first visited
$(document).ready(function(){

    matInfo[0] = { //all false by default
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
    $("#symm0").trigger("change");//blur out posdef. will also set the default root pc/ksp for the first time (see events.js)

    //display matrix pic. manually add square braces the first time
    //$("#matrixPic").html("<center>" + "\\(\\left[" + getMatrixTex("0") + "\\right]\\)" + "</center>");
    //MathJax.Hub.Queue(["Typeset",MathJax.Hub]);

    $(function() { //needed for jqueryUI tool tip to override native javascript tooltip
        $(document).tooltip();
    });

});