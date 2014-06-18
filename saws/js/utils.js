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
function getSawsIndex(id)
{
    for(var i=0; i<sawsInfo.length; i++) {
        if(sawsInfo[i].id == id)
            return i;//return index where information is located
    }
    return -1;//invalid id;
}

function getSawsDataIndex(id, endtag)//id is the adiv we are working with. endtag is the id of the data we are looking for
{
    if(id == -1)
        return -1;//invalid matrix id

    for(var i=0; i<sawsInfo[id].data.length; i++) {
        if(sawsInfo[id].data[i].endtag == endtag)
            return i;//return index where information is located
    }
    return -1;//invalid id;
}

//returns the index of that fieldsplit word in sawsInfo
function getFieldsplitWordIndex(word) {

    for(var i=0; i<sawsInfo.length; i++) {
        if(sawsInfo[i].name == word)
            return i;//return index where word was found
    }
    return -1;//word does not exist in sawsInfo yet
}

//this function iterates thru sawsInfo and returns the fieldsplit word at the given matrix id. if the id is invalid or the name does not exist, the function returns "-1". the function always returns a string.
function getSawsFieldsplitWord(id) {

    var index = getSawsIndex(id);
    if(index == -1)
        return "-1";//invalid index
    if(sawsInfo[index].name == undefined)
        return "-1";//given matrix does not have a name
    return sawsInfo[index].name;

}

//returns the ID of that fieldsplit word in sawsInfo
//this function always returns a string
function getFieldsplitWordID(word) {

    var index = getFieldsplitWordIndex(word);
    if(index == -1)
        return "-1";//word does not exist in sawsInfo yet
    return sawsInfo[index].id;
}

//count the number of children that current exist for the given parent
function getSawsNumChildren(parent) {

    var length = parent.length + 1;
    var count = 0;

    for(var i=0; i<sawsInfo.length; i++) {
        if(sawsInfo[i].id.length == length && sawsInfo[i].id.substring(0,sawsInfo[i].id.length-1) == parent)
            count++;
    }

    return count;
}