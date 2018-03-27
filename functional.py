#!/usr/bin/env python

from oamap.proxy import ListProxy
from vectorized import vectorize

# This is a hack to make Python's list and iterator types functional.

import ctypes
from functools import reduce

if hasattr(ctypes.pythonapi, "Py_InitModule4_64"):
    Py_ssize_t = ctypes.c_int64
else:
    Py_ssize_t = ctypes.c_int

class PyObject(ctypes.Structure): pass
PyObject._fields_ = [("ob_refcnt", Py_ssize_t), ("ob_type", ctypes.POINTER(PyObject))]

class SlotsPointer(PyObject):
    _fields_ = [("dict", ctypes.POINTER(PyObject))]

def proxy_builtin(cls):
    name = cls.__name__
    slots = getattr(cls, "__dict__", name)

    pointer = SlotsPointer.from_address(id(slots))
    namespace = {}

    ctypes.pythonapi.PyDict_SetItem(
        ctypes.py_object(namespace),
        ctypes.py_object(name),
        pointer.dict
    )

    return namespace[name]

#### Define and attach functional methods to the Python "list" type.

def sizer(lst):
    """
    Return the length of the list.
    
    Example: [1, 2, 3, 4, 5].size == 5
    
    (For convenience, since everything else is attached at the end of a chain of methods.)
    """
    return len(lst)

def enumerater(lst):
    """
    Return (index, element) pairs for all elemets of the list.

    Example: ["a", "b", "c", "d", "e"].enumerate ==
                 [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")]
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        return list(enumerate(lst))
    else:
        return ((i, x) for i, x in enumerate(lst))

def taker(lst):
    """
    Return the first n elements of the list or generator.
    
    Example: [1, 2, 3, 4, 5].take(3) == [1, 2, 3]
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        out = lambda n: lst[:n]
    else:
        def gen(n):
            for i, x in enumerate(lst):
                yield x
                if i + 1 >= n: break
        out = lambda n: list(gen(n))
    out.func_name = "[...].take"
    out.__doc__ = taker.__doc__
    return out

def collecter(lst):
    """
    Return all elements of the list or generator. This is a non-operation for lists, but evaluates generators (and might not halt if the generator is infinite).

    Example: (i for i in range(10)).collect == [i for i in range(10)]
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        return lst
    else:
        return list(lst)

def lazyer(lst):
    """
    Make a list or generator lazy. This is a non-operation for generators, which are already lazy, but makes it possible to build up a chain of operations on a list without evaluating them.

    Example: [1, 2, 3, 4, 5].lazy == (i for i in range(1, 6))
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        return (x for x in lst)
    else:
        return lst

def mapper(lst):
    """
    Apply a given function to each element of this list or generator.
    
    The function must take one argument.
    
    Examples: [1, 2, 3, 4, 5].map(f) == [f(1), f(2), f(3), f(4), f(5)]
              [1, 2, 3, 4, 5].map(lambda x: x + 100) == [101, 102, 103, 104, 105]
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        out = lambda f: [f(x) for x in lst]
    else:
        out = lambda f: (f(x) for x in lst)
    out.func_name = "[...].map"
    out.__doc__ = mapper.__doc__
    return out

def vmapper(lst):
    """
    Apply a given function to each element of this list or generator using vectorized function.
    
    The first argument must be a function, and the second argument must be an output array.
    Any number of argument can be passed to the vectorized function
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        # out = lambda f, len_out, *args: vectorize(f, len(lst), lst, *args)
        out = lambda f, out, *args: (vectorize(f, len(lst), out, lst, *args), out)[-1]
    else:
        out = lambda f, out, *args: (vectorize(f, len(lst), out, lst, *args), out)[-1]
    out.func_name = "[...].vmap"
    out.__doc__ = mapper.__doc__
    return out


def flattener(lst):
    """
    Turn a list-of-lists into a list of all elements. Only reduces one level of structure.
        
    Examples: [[1, 2], [3, 4, 5]].flatten == [1, 2, 3, 4, 5]
              [[1, 2], [3, [4, 5]]].flatten == [1, 2, 3, [4, 5]
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        return sum(lst, [])
    else:
        def gen():
            for x in lst:
                for y in x:
                    yield y
        return gen()

def flatmapper(lst):
    """
    Same as [...].map(f).flatten, but these two operations are frequently done together.
    
    The function must take one argument.
    
    In general: [...].flatmap(f) == [...].map(f).flatten
    
    Example: [1, 2, 3, 4, 5].flatmap(lambda x: [x, x + 100]) == [1, 101, 2, 102, 3, 103, 4, 104, 5, 105]
    
    Flatmap is a very general operation. You can use it to expand a table, as above, or to map and filter
    at the same time. (In the theory of monads, "flatmap" is the fundamental "bind" operation.)
    
    Example: [1, 2, 3, 4, 5].flatmap(lambda x: [100 + x] if x > 2 else []) == [103, 104, 105]
    
    You might encounter this when you want to compute something for all particles in each event, but also
    handle the case when there are no particles after cuts. In that case, "flatmap" instead of "map" and
    return a singleton list [result] when you have a result and an empty list [] when you don't.
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        out = lambda f: sum((f(x) for x in lst), [])
    else:
        def gen(f):
            for x in lst:
                for y in f(x):
                    yield y
        out = gen
    out.func_name = "[...].flatmap"
    out.__doc__ = flatmapper.__doc__
    return out

def filterer(lst):
    """
    Apply a given function to each element of the list and return only those that return True.
    
    The function must take one argument and return True or False.
    
    Example: [1, 2, 3, 4, 5].filter(lambda x: x > 2) == [3, 4, 5]
    """
    if isinstance(lst, (list, tuple, ListProxy)):
        out = lambda f: [x for x in lst if f(x)]
    else:
        def gen(f):
            for x in lst:
                if f(x):
                    yield x
        out = gen
    out.func_name = "[...].filter"
    out.__doc__ = filterer.__doc__
    return out
    
def reducer(lst):
    """
    Apply a given function to each element and a running tally to produce a single result.
    
    The function must take two arguments. The first may be an element from the list or a tally.
    The second will always be from the list.
    
    Examples: [1, 2, 3, 4, 5].reduce(f) == f(f(f(f(1, 2), 3), 4), 5)
              [1, 2, 3, 4, 5].reduce(lambda x, y: x + y) == 15
    """
    out = lambda f: reduce(f, lst)
    out.func_name = "[...].reduce"
    out.__doc__ = reducer.__doc__
    return out

def sumer(lst):
    """
    Compute the sum of all elements in the list.

    Example: [1, 2, 3, 4, 5].sum == 15
    """
    return sum(lst)

def aggregator(lst):
    """
    Same as reduce, except start the aggregation on a given zero element.
    
    The function must take two arguments. The first will always be a tally and the second from the list.
    
    Examples: [1, 2, 3, 4, 5].aggregate(f, 0) == f(f(f(f(f(0, 1), 2), 3), 4), 5)
              [1, 2, 3, 4, 5].aggregate(lambda x, y: x + y, 0) == 15
              ["a", "b", "c"].aggregate(lambda x, y: x + y, "") == "abc"
    """
    out = lambda f, zero: reduce(f, lst, zero)
    out.func_name = "[...].aggregate"
    out.__doc__ = aggregator.__doc__
    return out

def reducerright(lst):
    """
    Same as reduce, except start the nesting on the right and work left.
    
    The function must take two arguments. The second may be an element from the list or a tally.
    The first will always be from the list.
    
    Example: [1, 2, 3, 4, 5].reduceright(f) == f(1, f(2, f(3, f(4, 5))))
    """
    out = lambda f: reduce(lambda a, b: f(b, a), reversed(lst))
    out.func_name = "[...].reduceright"
    out.__doc__ = reducerright.__doc__
    return out

def aggregatorright(lst):
    """
    Same as aggregate, except start the nesting on the right and work left.
    
    The function must take two arguments. The second will always be a tally and the first from the list.
    
    Example: [1, 2, 3, 4, 5].aggregateright(f, 0) == f(1, f(2, f(3, f(4, f(5, 0)))))
    """
    out = lambda f, zero: reduce(lambda a, b: f(b, a), reversed(lst), zero)
    out.func_name = "[...].aggregateright"
    out.__doc__ = aggregatorright.__doc__
    return out

def pairser(lst):
    """
    Apply a given function to pairs of elements without repetition (in either order) or duplicates.
    
    The function must take two arguments. Both will always be elements from the list.
    
    If you think of the input list as a vector X, this acts on the upper trianglular part of the
    outer product of X with X (not including diagonal).
    
    Alternatively, it's what you would get from these nested loops:
    
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):   # j starts at i + 1
                f(lst[i], lst[j])
    
    Example: [1, 2, 3, 4, 5].pairs(lambda x, y: [x, y]) == [[1, 2], [1, 3], [1, 4], [1, 5],
                                                                    [2, 3], [2, 4], [2, 5],
                                                                            [3, 4], [3, 5],
                                                                                    [4, 5]]
    
    Use this when you want to loop over pairs of distinct pairs of elements from a single list.
    
    Contrast with "table", which is like a nested loop over several lists, for all elements.
    """
    out = lambda f: [f(x, y) for i, x in enumerate(lst) for y in lst[i + 1:]]
    out.func_name = "[...].pairs"
    out.__doc__ = pairser.__doc__
    return out

def tabler(lsts):
    """
    Apply a given function to all combinations of elements from all input lists.
    
    The function must take as many arguments as you have lists, and each will be an element from
    each list.
    
    If you think of the input lists as vectors X, Y, Z, etc., this acts on each element of the
    outer product of X with Y with Z, etc.
    
    Alternatively, it's what you would get from these nested loops:
    
        for x in lst_x:
            for y in lst_y:
                for z in lst_z:
                    f(x, y, z)
    
    Examples: [[100, 200], [1, 2, 3]].table(lambda x, y: x + y) == [101, 102, 103, 201, 202, 203]
    
              [[100, 200], [10, 20], [1, 2]].table(lambda x, y, z: x + y + z) == [
                  111, 112, 121, 122, 211, 212, 221, 222]

    To illustrate the difference between table and pairs, consider the following:

        [1, 2, 3].pairs(lambda x, y: [x, y]) == [[1, 2], [1, 3],
                                                         [2, 3]]
        
        [[1, 2, 3], [1, 2, 3]].table(lambda x, y: [x, y]) == [[1, 1], [1, 2], [1, 3],
                                                              [2, 1], [2, 2], [2, 3],
                                                              [3, 1], [3, 2], [3, 3]]
    """
    def buildargs(first, *rest):
        if len(rest) == 0:
            return [[x] for x in first]
        else:
            return [[x] + y for x in first for y in buildargs(*rest)]

    if len(lsts) == 0:
        out = lambda f: []
    elif len(lsts) == 1:
        out = lambda f: [f(x) for x in lsts[0]]
    else:
        first = lsts[0]
        rest = lsts[1:]
        out = lambda f: [f(*args) for args in buildargs(first, *rest)]
    out.func_name = "[[...], [...], ...].table"
    out.__doc__ = tabler.__doc__
    return out

def zipper(lsts):
    """
    Apply a function to the ith element of each list, for all i.
    
    The function must take as many arguments as there are lists, and each will be an element from
    each list.
    
    This works just like the built-in Python zip, but applies the function to its results:
    
        for x, y, z in zip(lst_x, lst_y, lst_z):
            f(x, y, z)
    
    Example: [[1, 2, 3], ["a", "b", "c"], [101, 102, 103]].zip(lambda x, y, z: (x, y, z)) == [
                 (1, "a", 101), (2, "b", 102), (3, "c", 103)]
    """
    if len(lsts) == 0:
        out = lambda f: []
    elif len(lsts) == 1:
        out = lambda f: [f(x) for x in lsts[0]]
    else:
        out = lambda f: [f(*args) for args in zip(*lsts)]
    out.func_name = "[[...], [...], ...].zip"
    out.__doc__ = zipper.__doc__
    return out

# attach the methods                                                force Python to notice
proxy_builtin(list)["size"] = property(sizer);                      hasattr([], "size")
proxy_builtin(list)["enumerate"] = property(enumerater);            hasattr([], "enumerate")
proxy_builtin(list)["take"] = property(taker);                      hasattr([], "take")
proxy_builtin(list)["collect"] = property(collecter);               hasattr([], "collect")
proxy_builtin(list)["lazy"] = property(lazyer);                     hasattr([], "lazy")
proxy_builtin(list)["map"] = property(mapper);                      hasattr([], "map")
proxy_builtin(list)["vmap"] = property(vmapper);                    hasattr([], "vmap")
proxy_builtin(list)["flatten"] = property(flattener);               hasattr([], "flatten")
proxy_builtin(list)["flatmap"] = property(flatmapper);              hasattr([], "flatmap")
proxy_builtin(list)["filter"] = property(filterer);                 hasattr([], "filter")
proxy_builtin(list)["reduce"] = property(reducer);                  hasattr([], "reduce")
proxy_builtin(list)["sum"] = property(sumer);                       hasattr([], "sum")
proxy_builtin(list)["aggregate"] = property(aggregator);            hasattr([], "aggregate")
proxy_builtin(list)["reduceright"] = property(reducerright);        hasattr([], "reduceright")
proxy_builtin(list)["aggregateright"] = property(aggregatorright);  hasattr([], "aggregateright")
proxy_builtin(list)["pairs"] = property(pairser);                   hasattr([], "pairs")
proxy_builtin(list)["table"] = property(tabler);                    hasattr([], "table")
proxy_builtin(list)["zip"] = property(zipper);                      hasattr([], "zip")

proxy_builtin(tuple)["size"] = property(sizer);                     hasattr((), "size")
proxy_builtin(tuple)["enumerate"] = property(enumerater);           hasattr((), "enumerate")
proxy_builtin(tuple)["take"] = property(taker);                     hasattr((), "take")
proxy_builtin(tuple)["collect"] = property(collecter);              hasattr((), "collect")
proxy_builtin(tuple)["lazy"] = property(lazyer);                    hasattr((), "lazy")
proxy_builtin(tuple)["map"] = property(mapper);                     hasattr((), "map")
proxy_builtin(tuple)["vmap"] = property(vmapper);                    hasattr([], "vmap")
proxy_builtin(tuple)["flatten"] = property(flattener);              hasattr((), "flatten")
proxy_builtin(tuple)["flatmap"] = property(flatmapper);             hasattr((), "flatmap")
proxy_builtin(tuple)["filter"] = property(filterer);                hasattr((), "filter")
proxy_builtin(tuple)["reduce"] = property(reducer);                 hasattr((), "reduce")
proxy_builtin(tuple)["sum"] = property(sumer);                      hasattr((), "sum")
proxy_builtin(tuple)["aggregate"] = property(aggregator);           hasattr((), "aggregate")
proxy_builtin(tuple)["reduceright"] = property(reducerright);       hasattr((), "reduceright")
proxy_builtin(tuple)["aggregateright"] = property(aggregatorright); hasattr((), "aggregateright")
proxy_builtin(tuple)["pairs"] = property(pairser);                  hasattr((), "pairs")
proxy_builtin(tuple)["table"] = property(tabler);                   hasattr((), "table")
proxy_builtin(tuple)["zip"] = property(zipper);                     hasattr((), "zip")

# also attach them to OAMap ListProxies
ListProxy.size = property(sizer)
ListProxy.enumerate = property(enumerater)
ListProxy.take = property(taker)
ListProxy.collect = property(collecter)
ListProxy.lazy = property(lazyer)
ListProxy.map = property(mapper)
ListProxy.flatten = property(flattener)
ListProxy.flatmap = property(flatmapper)
ListProxy.filter = property(filterer)
ListProxy.reduce = property(reducer)
ListProxy.sum = property(sumer)
ListProxy.aggregate = property(aggregator)
ListProxy.reduceright = property(reducerright)
ListProxy.aggregateright = property(aggregatorright)
ListProxy.pairs = property(pairser)
ListProxy.table = property(tabler)
ListProxy.zip = property(zipper)

# Verify that they all work (and provide examples of their use).

assert [1, 2, 3, 4, 5].take(3) == [1, 2, 3]

assert ["a", "b", "c", "d", "e"].enumerate == [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")]

assert [1, 2, 3, 4, 5].map(lambda x: 100 + x) == [101, 102, 103, 104, 105]

assert [[1, 2], [3, 4, 5]].flatten == [1, 2, 3, 4, 5]
assert [[1, 2], [3, [4, 5]]].flatten == [1, 2, 3, [4, 5]]

assert [1, 2, 3, 4, 5].map(lambda x: [x, x + 100]) == [[1, 101], [2, 102], [3, 103], [4, 104], [5, 105]]
assert [1, 2, 3, 4, 5].map(lambda x: [x, x + 100]).flatten == [1, 101, 2, 102, 3, 103, 4, 104, 5, 105]
assert [1, 2, 3, 4, 5].flatmap(lambda x: [x, x + 100]) == [1, 101, 2, 102, 3, 103, 4, 104, 5, 105]
assert [1, 2, 3, 4, 5].flatmap(lambda x: [100 + x] if x > 2 else []) == [103, 104, 105]

assert [1, 2, 3, 4, 5].filter(lambda x: x > 2) == [3, 4, 5]

assert [1, 2, 3, 4, 5].reduce(lambda x, y: x + y) == 15
assert [1, 2, 3, 4, 5].reduce(lambda x, y: [x, y]) == [[[[1, 2], 3], 4], 5]

assert ["a", "b", "c"].aggregate(lambda x, y: x + y, "") == "abc"
assert [1, 2, 3, 4, 5].aggregate(lambda x, y: [x, y], []) == [[[[[[], 1], 2], 3], 4], 5]

assert [1, 2, 3, 4, 5].reduceright(lambda x, y: [x, y]) == [1, [2, [3, [4, 5]]]]

assert [1, 2, 3, 4, 5].aggregateright(lambda x, y: [x, y], []) == [1, [2, [3, [4, [5, []]]]]]

assert [1, 2, 3, 4, 5].pairs(lambda x, y: [x, y]) == [[1, 2], [1, 3], [1, 4], [1, 5], [2, 3],
                                                      [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]

assert [[100, 200], [1, 2, 3]].table(lambda x, y: x + y) == [101, 102, 103, 201, 202, 203]
assert [[100, 200], [10, 20], [1, 2]].table(lambda x, y, z: x + y + z) == [111, 112, 121, 122, 211, 212, 221, 222]
assert [[1, 2, 3, 4, 5], ["a", "b"]].table(lambda x, y: [x, y]) == [
    [1, "a"], [1, "b"], [2, "a"], [2, "b"], [3, "a"], [3, "b"], [4, "a"], [4, "b"], [5, "a"], [5, "b"]]
assert [1, 2, 3].pairs(lambda x, y: [x, y]) == [[1, 2], [1, 3], [2, 3]]
assert [[1, 2, 3], [1, 2, 3]].table(lambda x, y: [x, y]) == [[1, 1], [1, 2], [1, 3],
                                                             [2, 1], [2, 2], [2, 3],
                                                             [3, 1], [3, 2], [3, 3]]

assert [[1, 2, 3], ["a", "b", "c"], [101, 102, 103]].zip(lambda x, y, z: [x, y, z]) == [
    [1, "a", 101], [2, "b", 102], [3, "c", 103]]

def example_generator():
    yield None

example_generator = example_generator()

proxy_builtin(type(example_generator))["enumerate"] = property(enumerater); hasattr(example_generator, "enumerate")
proxy_builtin(type(example_generator))["take"] = property(taker);           hasattr(example_generator, "take")
proxy_builtin(type(example_generator))["collect"] = property(collecter);    hasattr(example_generator, "collect")
proxy_builtin(type(example_generator))["lazy"] = property(lazyer);          hasattr(example_generator, "lazy")
proxy_builtin(type(example_generator))["map"] = property(mapper);           hasattr(example_generator, "map")
proxy_builtin(type(example_generator))["flatten"] = property(flattener);    hasattr(example_generator, "flatten")
proxy_builtin(type(example_generator))["flatmap"] = property(flatmapper);   hasattr(example_generator, "flatmap")
proxy_builtin(type(example_generator))["filter"] = property(filterer);      hasattr(example_generator, "filter")
