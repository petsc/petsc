/*
  getMatIndex -
  input:
    desired id in string format. (for example, "01001")
  output:
    index in matInfo where information on that id is located
*/
function getMatIndex(id)
{
    for (var i=0; i<matInfo.length; i++) {
        if (matInfo[i].id == id)
            return i;//return index where information is located.
    }
    return -1;//invalid id.
}

/*
  getSawsIndex -
  input:
    desired id in string format. (for example, "01001")
  output:
    index in sawsInfo where information on that id is located
*/

function getSawsIndex(endtag) {

    for(var i=0; i<sawsInfo.length; i++) {
        if(sawsInfo[i].endtag == endtag)
            return i;//return index where information is located
    }
    return -1;//invalid endtag;
}

//return the index for the given fieldsplit name (if any)
function getFieldsplitWordIndex(word) {

    for(var i=0; i<sawsInfo.length; i++) {
        if(sawsInfo[i].name == word)
            return i;//return index where word was found
    }
    return -1;//word does not exist in sawsInfo yet
}

//count the number of children that current exist for the given parent
function getSawsNumChildren(parent) {

    var childNumUnderscores = getNumUnderscores(parent) + 1;
    var count               = 0;

    for(var i=0; i<sawsInfo.length; i++) {
        if(getNumUnderscores(sawsInfo[i].endtag) == childNumUnderscores && sawsInfo[i].endtag.substring(0,sawsInfo[i].endtag.lastIndexOf('_')) == parent) //if child level is the same, and parent matches...
            count++;
    }

    return count;
}

//returns the number of underscores in the endtag
function getNumUnderscores(endtag) {

    var count = 0;
    for(var i=0; i<endtag.length; i++) {
        if(endtag.charAt(i) == "_")
            count ++;
    }
    return count;
}

//returns the endtag of the parent (if any)
function getParent(endtag) {

    if(endtag.indexOf('_') == -1)
        return "-1"; //has no parent or invalid endtag

    return endtag.substring(0,endtag.lastIndexOf('_'));
}