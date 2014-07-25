//gets the index of the given endtag in the data array
function getIndex(data,endtag)
{
    for (var i=0; i<data.length; i++) {
        if (data[i].endtag == endtag)
            return i;//return index where information is located.
    }
    return -1;//not found
}

//gets the index of the given name in the data array. must also specify parent because we allow same fieldsplit names under different parents
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

//returns the index of the parent (if any)
function getParentIndex(data,endtag) {

    var parentEndtag = getParent(endtag);
    if(parentEndtag == "-1")
        return -1;

    return getIndex(data,parentEndtag);
}

//returns the number of occurances of a string in another string
function countNumOccurances(small_string, big_string) {

    var count = 0;

    while(small_string.length <= big_string.length && big_string.indexOf(small_string) != -1) {
        count ++;
        var loc = big_string.indexOf(small_string);
        big_string = big_string.substring(loc + small_string.length, big_string.length);
    }

    return count;
}
