def get_domain(whole_graph):
    loc_set = set()
    for c_idx in whole_graph['cnt_node_idx']:
        loc_set.add(whole_graph['node_names'][c_idx])
    loc_str = ''
    for loc in loc_set:
        loc_str += ' ' + loc

    obj_set = set()
    for o_idx in whole_graph['obj_node_idx']:
        obj_set.add(whole_graph['node_names'][o_idx])
    obj_str = ''
    for obj in obj_set:
        obj_str += ' ' + obj

    DOMAIN_PDDL = f"""
    (define
    (domain indoor)

    (:requirements :strips :typing :action-costs :existential-preconditions)

    (:types
        location item - object
        init_r{loc_str} - location
        {obj_str} - item
    )

    (:predicates
        (is-holding ?obj - item)
        (is-located ?obj - item)
        (is-at ?obj - item ?loc - location)
        (rob-at ?loc - location)
        (hand-is-free)
        (is-pickable ?obj - item)
        (restrict-move-to ?loc - location)
        (is-toasted ?obj - item)
        (is-toastable ?obj - item)
        (is-toaster ?obj - item)
        (is-peeled ?obj - item)
        (is-peelable ?obj - item)
        (is-peeler ?obj - item)
        (is-boiled ?obj - item)
        (is-boilable ?obj - item)
        (is-boiler ?obj - item)
        (ban-move)
    )

    (:functions
        (known-cost ?start ?end)
        (find-cost ?obj ?loc)
        (total-cost)
    )

    (:action boil
        :parameters (?boilitem - item ?boiler - item ?loc - location)
        :precondition (and
            (hand-is-free)
            (not (is-boiled ?boilitem))
            (is-boilable ?boilitem)
            (is-at ?boilitem ?loc)
            (is-boiler ?boiler)
            (is-at ?boiler ?loc)
            (rob-at ?loc)
        )
        :effect (and
            (is-boiled ?boilitem)
            (not (ban-move))
            (increase (total-cost) 100)
        )
    )

    (:action peel
        :parameters (?peelitem - item ?peeler - item ?loc - location)
        :precondition (and
            (not (is-peeled ?peelitem))
            (is-peelable ?peelitem)
            (is-at ?peelitem ?loc)
            (is-holding ?peeler)
            (is-peeler ?peeler)
            (rob-at ?loc)
        )
        :effect (and
            (is-peeled ?peelitem)
            (not (ban-move))
            (increase (total-cost) 100)
        )
    )

    (:action toast
        :parameters (?toastitem - item ?toaster - item ?loc - location)
        :precondition (and
            (hand-is-free)
            (not (is-toasted ?toastitem))
            (is-toastable ?toastitem)
            (is-at ?toastitem ?loc)
            (is-toaster ?toaster)
            (is-at ?toaster ?loc)
            (rob-at ?loc)
        )
        :effect (and
            (is-toasted ?toastitem)
            (not (ban-move))
            (increase (total-cost) 100)
        )
    )

    (:action pick
        :parameters (?obj - item ?loc - location)
        :precondition (and
            (is-pickable ?obj)
            (is-located ?obj)
            (is-at ?obj ?loc)
            (rob-at ?loc)
            (hand-is-free)
        )
        :effect (and
            (not (is-at ?obj ?loc))
            (is-holding ?obj)
            (not (hand-is-free))
            (not (ban-move))
            (increase (total-cost) 100)
        )
    )

    (:action place
        :parameters (?obj - item ?loc - location)
        :precondition (and
            (not (hand-is-free))
            (rob-at ?loc)
            (is-holding ?obj)
        )
        :effect (and
            (is-at ?obj ?loc)
            (not (is-holding ?obj))
            (hand-is-free)
            (not (ban-move))
            (increase (total-cost) 100)
        )
    )

    (:action move
        :parameters (?start - location ?end - location)
        :precondition (and
            (not (= ?start ?end))
            (not (restrict-move-to ?end))
            (not (ban-move))
            (rob-at ?start)
        )
        :effect (and
            (not (rob-at ?start))
            (rob-at ?end)
            (ban-move)
            (increase (total-cost) (known-cost ?start ?end))
        )
    )

    (:action find
        :parameters (?obj - item ?from - location ?to - location)
        :precondition (and
            (not (restrict-move-to ?to))
            (rob-at ?from)
            (not (is-located ?obj))
            (is-pickable ?obj)
            (hand-is-free)
        )
        :effect (and
            (is-located ?obj)
            (not (hand-is-free))
            (is-holding ?obj)
            (rob-at ?to)
            (not (rob-at ?from))
            (not (ban-move))
            (increase (total-cost) (find-cost ?obj ?from ?to))
        )
    )

    )
    """
    return DOMAIN_PDDL
