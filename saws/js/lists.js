//populate fieldsplit list
function populateFieldsplitList(listId) 
{
    $("#" + listId).append("<option value=\"multiplicative\">multiplicative</option>")
    $("#" + listId).append("<option value=\"additive\">additive</option>")
    $("#" + listId).append("<option value=\"symmetric_multiplicative\">symmetric_multiplicative</option>")
    $("#" + listId).append("<option value=\"special\">special</option>")
    $("#" + listId).append("<option value=\"schur\">schur</option>")
}

//populate mg list
function populateMgList(listId) 
{
    $("#" + listId).append("<option value=\"multiplicative\">multiplicative</option>")
    $("#" + listId).append("<option value=\"additive\">additive</option>")
    $("#" + listId).append("<option value=\"full\">full</option>")
    $("#" + listId).append("<option value=\"kaskade\">kaskade</option>")
}

/*
  populateKspList - populate the ksp list
  input
    listId - string
    listVals - an array of ksptypes, e.g. listVals = ["cg","(null)"]; 
    defaultVal - default pctype
*/
function populateKspList(listId,listVals,defaultVal)
{
    var list = "#" + listId;
    $(list).empty(); //empty existing list
    //alert("enter  populateKspList...");
    
    if (listVals == null) {
        //all options without parenthesis are for nonsymmetric (and, therefore, non pd) KSP list
        $(list).append("<option value=\"bcgs\">bcgs</option>");
        $(list).append("<option value=\"bcgsl\">bcgsl</option>");
        $(list).append("<option value=\"bicg\">bicg</option>");
        $(list).append("<option value=\"cg\">cg (symm, positive definite)</option>");
        $(list).append("<option value=\"cgne\">cgne</option>");
        $(list).append("<option value=\"cgs\">cgs</option>");
        $(list).append("<option value=\"chebyshev\">chebyshev</option>");
        $(list).append("<option value=\"cr\">cr</option>");
        $(list).append("<option value=\"fgmres\">fgmres</option>");
        $(list).append("<option value=\"gltr\">gltr</option>");
        $(list).append("<option value=\"gmres\">gmres</option>");
        $(list).append("<option value=\"groppcg\">groppcg</option>");
        $(list).append("<option value=\"lsqr\">lsqr</option>");
        $(list).append("<option value=\"minres\">minres (symm, non-positive definite)</option>");
        $(list).append("<option value=\"nash\">nash</option>");
        $(list).append("<option value=\"pgmres\">pgmres</option>");
        $(list).append("<option value=\"pipecg\">pipecg</option>");
        $(list).append("<option value=\"pipecr\">pipecr</option>");
        $(list).append("<option value=\"preonly\">preonly</option>");
        $(list).append("<option value=\"qcg\">qcg (symm, positive definite)</option>");
        $(list).append("<option value=\"richardson\">richardson</option>");
        $(list).append("<option value=\"specest\">specest</option>");
        $(list).append("<option value=\"stcg\">stcg</option>");
        $(list).append("<option value=\"symmlq\">symmlq (symm, non-positive definite)</option>");
        $(list).append("<option value=\"tcqmr\">tcqmr</option>");
        $(list).append("<option value=\"tfqmr\">tfqmr</option>");
    } else {
        var i=0;
        while (listVals[i] != "(null)"){
            $(list).append("<option value="+listVals[i]+">"+listVals[i]+"</option>");
            i++;
        }
    }

    //set defaults ksp_type
    if (defaultVal != "null") {
        $(list).find("option[value=" + defaultVal +"]").attr("selected","selected"); 
    } else {
        var recursionCounter = oDivGetRecursionCounter(listId);
        if (matInfo[recursionCounter].symm == undefined) {
            alert("Warning: matInfo["+recursionCounter+"].symm is undefined!");
            $(list).find("option[value='gmres']").attr("selected","selected");
        } else if (matInfo[recursionCounter].symm && !matInfo[recursionCounter].posdef) {
	    $(list).find("option[value='minres']").attr("selected","selected");
        } else if (matInfo[recursionCounter].symm && matInfo[recursionCounter].posdef) {
	    $(list).find("option[value='cg']").attr("selected","selected");
        } else {
	    $(list).find("option[value='gmres']").attr("selected","selected");
        }
    }
}

/*
  populatePcList - populate the pc list
  input:
    listId - string
    listVals - an array of pctypes, e.g. listVals = ["none","jacobi","pbjacobi","bjacobi","(null)"]; the laster entry must be "(null)"
    defaultVal - default pctype
*/
function populatePcList(listId,listVals,defaultVal) 
{
    var list="#"+listId;
    $(list).empty(); //empty existing list

    if (listVals == null) {
        $(list).append("<option value=\"asa\">asa</option>");
        $(list).append("<option value=\"asm\">asm</option>");
        $(list).append("<option value=\"bjacobi\">bjacobi</option>");
        $(list).append("<option value=\"cholesky\">cholesky</option>");
        $(list).append("<option value=\"composite\">composite</option>");
        $(list).append("<option value=\"cp\">cp</option>");
        $(list).append("<option value=\"eisenstat\">eisenstat</option>");
        $(list).append("<option value=\"exotic\">exotic</option>");
        $(list).append("<option value=\"fieldsplit\">fieldsplit (block structured)</option>");
        $(list).append("<option value=\"galerkin\">galerkin</option>");
        $(list).append("<option value=\"gamg\">gamg</option>");
        $(list).append("<option value=\"gasm\">gasm</option>");
        $(list).append("<option value=\"hmpi\">hmpi</option>");
        $(list).append("<option value=\"icc\">icc</option>");
        $(list).append("<option value=\"ilu\">ilu</option>");
        $(list).append("<option value=\"jacobi\">jacobi</option>");
        $(list).append("<option value=\"ksp\">ksp</option>");
        $(list).append("<option value=\"lsc\">lsc</option>");
        $(list).append("<option value=\"lu\">lu</option>");
        $(list).append("<option value=\"mat\">mat</option>");
        $(list).append("<option value=\"mg\">mg</option>");
        $(list).append("<option value=\"nn\">nn</option>");
        $(list).append("<option value=\"none\">none</option>");
        $(list).append("<option value=\"pbjacobi\">pbjacobi</option>");
        $(list).append("<option value=\"redistribute\">redistribute</option>");
        $(list).append("<option value=\"redundant\">redundant</option>");
        $(list).append("<option value=\"shell\">shell</option>");
        $(list).append("<option value=\"sor\">sor</option>");
        $(list).append("<option value=\"svd\">svd</option>");
    } else {
        var i=0;
        while (listVals[i] != "(null)"){
            $(list).append("<option value="+listVals[i]+">"+listVals[i]+"</option>");
            i++;
        }
    }

    //set default pc_type
    var recursionCounter = oDivGetRecursionCounter(listId);
    if (matInfo[recursionCounter].logstruc == undefined) {
        alert("Warning: matInfo["+recursionCounter+"].logstruc is undefined!");
        if (defaultVal == "null") {
	    $(list).find("option[value='bjacobi']").attr("selected","selected");  
        } else {
            $(list).find("option[value=" + defaultVal +"]").attr("selected","selected"); 
        }
    } else if (matInfo[recursionCounter].logstruc) {
	$(list).find("option[value='fieldsplit']").attr("selected","selected");
    } else { //!matInfo[recursionCounter].logstruc
        if (defaultVal == "null") {
	    $(list).find("option[value='bjacobi']").attr("selected","selected");  
        } else {
            $(list).find("option[value=" + defaultVal +"]").attr("selected","selected"); 
        }
    }
}

/*
  oDivGetRecursionCounter - get recursionCounter of oDiv that contains the input object (copied from listLogic.js)
  input:
    objId
  output:
    recursionCounter of oDiv that contains the input object
*/
function oDivGetRecursionCounter(objId) 
{
    var parentDiv = $("#"+objId).parent().get(0).id;
    var recursionCounter = parentDiv;
    //alert('recursionCounter='+recursionCounter);
    if (recursionCounter == "") {
        recursionCounter = -1;
    } else {
        while (recursionCounter.indexOf('_') != -1)
	    recursionCounter=$("#"+recursionCounter).parent().get(0).id;
        recursionCounter = recursionCounter.substring(1, recursionCounter.length);
    }
    return recursionCounter;
}