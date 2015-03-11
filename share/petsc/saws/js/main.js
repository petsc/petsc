//this data structure is used to hold all of the solver options and matrix properties (perhaps should use two separate ones?)
var matInfo = {};

//some global boolean variables to keep track of what the user wants to display
var displayCmdOptions = true;
var displayTree       = true;
var displayMatrix     = true;
//var displayDiagram    = true;

//holds the cmd options to copy to the clipboard
var clipboardText     = "";

//holds the colors used in the diagram drawing
var colors = ["black","red","blue","green"];

//  This function is run when the page is first visited
$(document).ready(function(){

    matInfo["0"] = { //all false by default
        logstruc: false,
        symm: false,
        posdef: false,
    };

    //to start, append the first div (div0) in the table and the first pc/ksp options dropdown
    $("#results").append("<div id=\"leftPanel\" style=\"background-color:lightblue;float:left;\"> </div> <div id=\"rightPanel\" style=\"float:left;padding-left:30px;\"></div>");
    $("#leftPanel").append("<div id=\"solver0\"> </div><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>");

    $("#solver0").append("<b>Root Solver Options (Matrix is <input type=\"checkbox\" id=\"symm0\">symmetric, <input type=\"checkbox\" id=\"posdef0\">positive definite, <input type=\"checkbox\" id=\"logstruc0\">block structured)</b>");//text: Solver Level: 0
    $("#solver0").append("<br><b>KSP &nbsp;</b><select id=\"ksp_type0\"></select>");
    $("#solver0").append("<br><b>PC &nbsp;&nbsp;&nbsp;</b><select id=\"pc_type0\"></select>");

    populateList("pc","0");
    populateList("ksp","0");

    $("#pc_type0").trigger("change");//display options for sub-solvers (if any)
    $("#ksp_type0").trigger("change");//just to record ksp (see listLogic.js)
    $("#symm0").trigger("change");//blur out posdef. will also set the default root pc/ksp for the first time (see events.js)

    /* $(function() { //needed for jqueryUI tool tip to override native javascript tooltip
        $(document).tooltip();
    });*/

    $("#displayCmdOptions").attr("checked",true);
    $("#displayTree").attr("checked",true);
    $("#displayMatrix").attr("checked",true);
    //$("#displayDiagram").attr("checked",true);

});
