import argparse
import pickle

from pythomata.impl.symbolic import SymbolicDFA

if __name__ == '__main__':
    parser = argparse.ArgumentParser("fix-automaton")
    parser.add_argument("-i", type=str, help="Path to automaton pickle file.", default="inputs/automaton.pickle")
    parser.add_argument("-o", type=str, help="Path to output file.", default="output.pickle")
    parser.add_argument("--render", type=str, help="Path to dot file.", default=None)

    arguments = parser.parse_args()
    with open(arguments.i, "rb") as fin:
        automaton: SymbolicDFA = pickle.load(fin)

    # States numbers
    to_explore = [automaton.initial_state]
    next_id = 0
    new_ids = {}
    while len(to_explore) > 0:
        state = to_explore.pop(0)
        new_ids[state] = next_id
        next_id += 1
        for successor in automaton._transition_function[state].keys():
            if successor not in new_ids:
                to_explore.append(successor)

    # New automaton and states
    new_automaton = SymbolicDFA()
    for old_state in new_ids:
        if automaton.initial_state == old_state:
            new_state = new_ids[old_state]
            assert new_ids[old_state] == 0
        else:
            new_state = new_automaton.create_state()
            assert new_state == new_ids[old_state]

        new_automaton.set_accepting_state(
            new_state, automaton.is_accepting(old_state))

    # Transitions
    for old_state in automaton.states:
        for old_next, guard in automaton._transition_function[old_state].items():
            new_automaton.add_transition((
                new_ids[old_state], guard, new_ids[old_next]
            ))

    # Save
    if arguments.render is not None:
        new_automaton.to_graphviz().render(arguments.render)
    with open(arguments.o, "wb") as fout:
        pickle.dump(new_automaton, fout, protocol=4)
