def pour_water(container=None):
    if container:
        str = f'(filled-with water {container})'
    else:
        str = '''
        (exists
            (?c - item)
                (and (filled-with water ?c))
        )
        '''
    return str


def serve_water(location, container=None):
    if container:
        str = f'(and (filled-with water {container}) (is-at {container} {location}))'
    else:
        str = f'''
        (exists
            (?c - item)
                (and (filled-with water ?c) (is-at ?c {location}))
        )
        '''
    return str


def make_coffee(container=None):
    if container:
        str = f'(filled-with coffee {container})'
    else:
        str = '''
        (exists
            (?c - item)
                (and (filled-with coffee ?c))
        )
        '''
    return str


def serve_coffee(location, container=None):
    if container:
        str = f'(and (filled-with coffee {container}) (is-at {container} {location}))'
    else:
        str = f'''
        (exists
            (?c - item)
                (and (filled-with coffee ?c) (is-at ?c {location}))
        )
        '''
    return str


def clean_something(object):
    str = f'(not (is-dirty {object}))'
    return str


def make_sandwich(spread=None):
    if spread:
        str = f'(spread-applied bread {spread})'
    else:
        str = '''
        (exists
            (?sprd - spread)
                (spread-applied bread ?sprd)
        )
        '''
    return str


def serve_sandwich(location, spread=None):
    if spread:
        str = f'(and (spread-applied bread {spread}) (is-at bread {location}))'
    else:
        str = f'''
        (exists
            (?sprd - spread)
                (and (spread-applied bread ?sprd) (is-at bread {location}))
        )
        '''
    return str


def set_napkin(location, napkin=None):
    if napkin:
        str = f'(and (is-folded {napkin}) (is-at {napkin} {location}))'
    else:
        str = f'''
        (exists (?npkn - napkin)
            (and (is-folded ?npkn) (is-at ?npkn {location}))
        )
        '''
    return str


def place_one_object(locs, objs):
    # t_str = f'(is-at {objs[0]} {locs[0]})'
    # get generic name of the object
    gen_obj = [obj.split('|')[0] for obj in objs]
    t_str = f'(exists (?obj - item) (and (is-at ?obj {locs[0]}) (obj-type-{gen_obj[0]} ?obj)))'
    return t_str


def place_two_objects(locs, objs):
    # t_str = f'(and (is-at {objs[0]} {locs[0]}) (is-at {objs[1]} {locs[1]}))'
    gen_obj = [obj.split('|')[0] for obj in objs]
    t_str1 = f'(exists (?obj - item) (and (is-at ?obj {locs[0]}) (obj-type-{gen_obj[0]} ?obj)))'
    t_str2 = f'(exists (?obj - item) (and (is-at ?obj {locs[1]}) (obj-type-{gen_obj[1]} ?obj)))'
    t_str = f'(and {t_str1} {t_str2})'
    return t_str


def place_three_objects(locs, objs):
    # t_str = f'(and (is-at {objs[0]} {locs[0]}) (is-at {objs[1]} {locs[1]}) (is-at {objs[2]} {locs[2]}))'
    gen_obj = [obj.split('|')[0] for obj in objs]
    t_str1 = f'(exists (?obj - item) (and (is-at ?obj {locs[0]}) (obj-type-{gen_obj[0]} ?obj)))'
    t_str2 = f'(exists (?obj - item) (and (is-at ?obj {locs[1]}) (obj-type-{gen_obj[1]} ?obj)))'
    t_str3 = f'(exists (?obj - item) (and (is-at ?obj {locs[2]}) (obj-type-{gen_obj[2]} ?obj)))'
    t_str = f'(and {t_str1} {t_str2} {t_str3})'
    return t_str


def multiple_goal(goals):
    t_str = '(or'
    for goal in goals:
        t_str += f' {goal}\n'
    t_str += ')'
    return t_str


def get_related_goal(goal_cnt, goal_objs, combine=True):
    obj1 = goal_objs[1].split('|')[0]
    obj2 = goal_objs[0].split('|')[0]
    if obj1 == 'egg':
        state_str = '(is-boiled ?obj1)'
    elif obj1 in ['apple', 'tomato', 'potato']:
        state_str = '(is-peeled ?obj1)'
    elif obj1 == 'bread':
        state_str = '(is-toasted ?obj1)'
    # obj1_loc_str = f'(is-at {goal_objs[1]} {goal_cnt})'
    obj1_loc_str = f'(exists (?obj1 - item) (and (obj-type-{obj1} ?obj1) {state_str} (is-at ?obj1 {goal_cnt})))'
    obj2_loc_str = f'(exists (?obj2 - item) (and (obj-type-{obj2} ?obj2) (is-at ?obj2 {goal_cnt})))'
    if not combine:
        return f'{obj1_loc_str} {obj2_loc_str}'
    combined_str = f'(and {obj1_loc_str} {obj2_loc_str})'
    return combined_str


def get_coffee_task(goal_cnt, goal_obj, combine=True):
    gen_obj = goal_obj.split('|')[0]
    state_str = '(filled-with-coffee ?obj)'
    obj_loc_str = f'(is-at ?obj {goal_cnt})'
    t_str = f'(exists (?obj - item) (and (obj-type-{gen_obj} ?obj) {state_str} {obj_loc_str}))'
    if not combine:
        return f'{t_str}'
    combined_str = f'(and {t_str})'
    return combined_str
