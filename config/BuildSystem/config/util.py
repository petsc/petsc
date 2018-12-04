# Stateless utilities
#
# This entire module is exported into the config.* namespace (see
# __init__.py), so be careful to avoid namespace conflicts.  Do not
# import config.util directly.  Instead, import config and access only
# the exported objects.

def classify(items, functional, args=(), kwargs=dict()):
    '''Classify items as True or False using boolean functional on sets.

    An item can only be identified as False if functional([item]) is
    false, but an item is True if functional(List) is True and item in
    List.

    Functional may return True (all items are True), False (at least one
    item is False) or a list of suggestions for items that may have been
    False (implies that at least one item is False).  The list of
    suggestions does not have to be accurate, but classification will be
    faster if it is accurate.
    '''
    items = list(items)         # In case a set or other iterable was passed in

    result = functional(items, *args, **kwargs)
    if result is True:
        return items, []        # All succeeded
    if len(items) == 1:
        return [], items        # Sole failure
    if result is False:
        suggested = []
    else:
        suggested = list(result)
    items = [i for i in items if i not in suggested]
    good = []
    bad = []
    if len(items) < 5:          # linear check
        groups = [[i] for i in items]
    else:                       # bisect
        groups = [items[:len(items)//2], items[len(items)//2:]]
    groups += [[i] for i in suggested]
    for grp in groups:
        g, b = classify(grp, functional, args, kwargs)
        good += g
        bad += b
    return good, bad


class NamedInStderr:
    '''Hepler class to log the (string) items that are written to stderr on failure.

    In the common case, all the missing items are named in the linker
    error and the rest can be confirmed True in a single batch.
    '''
    def __init__(self, items):
        self.named = []
        self.items = items

    def examineStderr(self, ret, out, err):
        if ret:
            self.named += [i for i in self.items if i in err]


class memoize(dict):
    '''Memoizing decorator.  No support for keyword arguments.'''
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        try:
            return self[args]
        except KeyError:
            ret = self[args] = self.func(*args)
            return ret
