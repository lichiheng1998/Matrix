

def map_2d(func, row, col, *args, construct=None,
           iter_key=None, assign_key=None):
    def assign_func(container, r, c, value):
        container[(r, c)] = value

    if assign_key is None:
        assign_key = assign_func
    if construct is None:
        construct = {}
    return _my_map2_rec(func, row, col, *args,
                        r=0,
                        c=0,
                        construct=construct,
                        assign_func=assign_key,
                        iter_func=iter_key
                        )


def _my_map2_rec(func, row, col, *args, r, c, construct, iter_func,
                 assign_func):
    value_list = tuple(it(item, r, c) for (item, it) in zip(args, iter_func))
    value = func(*value_list)
    assign_func(construct, r, c, value)
    if r == row - 1 and c == col - 1:
        return construct
    (r, c) = (r + 1, 0) if c == col - 1 else (r, c + 1)
    return _my_map2_rec(func, row, col, *args, r=r, c=c, construct=construct,
                        iter_func=iter_func, assign_func=assign_func)
