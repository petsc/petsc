function getIndex(data,endtag)
{
    for (var i=0; i<data.length; i++) {
        if (data[i].endtag == endtag)
            return i;//return index where information is located.
    }
    return -1;//not found
}

function getIndexByName(data,name,parent) {

    for(var i=0; i<data.length; i++) {
        if(data[i].name == name && data[i].endtag.indexOf(parent) == 0)
            return i;//return index where the name was found
    }
    return -1;//not found
}

//count the number of children that current exist for the given parent
function getNumChildren(data,parent) {

    var childNumUnderscores = getNumUnderscores(parent) + 1;
    var count               = 0;

    for(var i=0; i<data.length; i++) {
        if(getNumUnderscores(data[i].endtag) == childNumUnderscores && data[i].endtag.substring(0,data[i].endtag.lastIndexOf('_')) == parent) //if child level is the same, and parent matches...
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
        return "-1"; //has no parent (root case) or invalid endtag

    return endtag.substring(0,endtag.lastIndexOf('_'));
}





































/*
  getMatIndex -
  input:
    desired id in string format. (for example, "01001")
  output:
    index in matInfo where information on that id is located
*/
//function getMatIndex(id)
//{
//    for (var i=0; i<matInfo.length; i++) {
//        if (matInfo[i].id == id)
//            return i;//return index where information is located.
//    }
//    return -1;//invalid id.
//}

/*
  getSawsIndex -
  input:
    desired id in string format. (for example, "01001")
  output:
    index in sawsInfo where information on that id is located
*/

/*function getSawsIndex(endtag) {

    for(var i=0; i<sawsInfo.length; i++) {
        if(sawsInfo[i].endtag == endtag)
            return i;//return index where information is located
    }
    return -1;//invalid endtag;
}*/

/*

//return the index for the given fieldsplit name (if any)
//we need to specify the parent as well because we allow fieldsplits with the same name as long as they are under different parents. this actually happens quite often with unnamed fieldsplits that are given a numerical name by default.
function getFieldsplitWordIndex(word,parent) {

    for(var i=0; i<sawsInfo.length; i++) {
        if(sawsInfo[i].name == word && sawsInfo[i].endtag.indexOf(parent) == 0)
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

*/