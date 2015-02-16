//gets the endtag of the given name in the data object. must also specify parent because we allow same fieldsplit names under different parents
function getEndtagByName(data,name,parent) {

    for (var key in data) {
        if (data.hasOwnProperty(key)) {
            if(data[key].name == name && key.indexOf(parent) == 0)
                return key;
        }
    }
    return "-1";
}

//count the number of children that currently exist for the given parent
function getNumChildren(data,parent) {

    var childNumUnderscores = getNumUnderscores(parent) + 1;
    var count               = 0;

    for (var key in data) {
        if (data.hasOwnProperty(key)) {
            if(getNumUnderscores(key) == childNumUnderscores && key.indexOf(parent) == 0)
                count ++;
        }
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

//scrolls the page
function scrollTo(id)
{
    $('html,body').animate({scrollTop: $("#"+id).offset().top},'fast');
}
