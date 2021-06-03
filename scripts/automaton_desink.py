import argparse
import pickle
from copy import deepcopy

from pythomata.impl.symbolic import SymbolicDFA

if __name__ == '__main__':
    parser = argparse.ArgumentParser("fix-automaton")
    parser.add_argument("-i", type=str, help="Path to automaton pickle file.", default="inputs/automaton.pickle")
    parser.add_argument("-o", type=str, help="Path to output file.", default="output.pickle")
    parser.add_argument("--render", type=str, help="Path to dot file.", default=None)

    arguments = parser.parse_args()
    with open(arguments.i, "rb") as fin:
        automaton = pickle.load(fin)

    automaton: SymbolicDFA

    failure_state = 11
    initial_state = 0
    transition_function = deepcopy(automaton._transition_function)

    for start_state, out_transitions in transition_function.items():
        if failure_state in out_transitions:
            guard = automaton._transition_function[start_state].pop(failure_state)
            automaton.add_transition((start_state, guard, initial_state))

    automaton.states.remove(failure_state)
    if arguments.render is not None:
        automaton.to_graphviz().render(arguments.render)
    with open(arguments.o, "wb") as fout:
        pickle.dump(automaton, fout, protocol=4)
