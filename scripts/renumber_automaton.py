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
    new2old = []
    while len(to_explore) > 0:
        state = to_explore.pop(0)
        new2old.append(state)
        for successor in sorted(automaton._transition_function[state].keys()):
            if successor not in new2old and successor not in to_explore:
                to_explore.append(successor)
    old2new = {old: new for new, old in enumerate(new2old)}

    # New automaton and states
    new_automaton = SymbolicDFA()
    for new_state, old_state in enumerate(new2old):
        if automaton.initial_state == old_state:
            assert old_state == 0
        else:
            new_assigned_state = new_automaton.create_state()
            assert new_assigned_state == new_state

        new_automaton.set_accepting_state(
            new_state, automaton.is_accepting(old_state))

    # Transitions
    for old_state in automaton.states:
        for old_next, guard in automaton._transition_function[old_state].items():
            new_automaton.add_transition((
                old2new[old_state], guard, old2new[old_next]
            ))

    # Save
    if arguments.render is not None:
        new_automaton.to_graphviz().render(arguments.render)
    with open(arguments.o, "wb") as fout:
        pickle.dump(new_automaton, fout, protocol=4)
