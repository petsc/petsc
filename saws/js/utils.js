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
    for(var i=0; i<sawsInfo[id].data.length; i++) {
        if(sawsInfo[id].data[i].endtag == endtag)
            return i;//return index where information is located
    }
    return -1;//invalid id;
}


function getFieldsplitWordIndex(word) {

    for(var i=0; i<fieldsplitKeywords.length; i++) {
        if(fieldsplitKeywords[i] == word)
            return i;//return index where word was found
    }
    return -1;//word does not exist in array yet
}