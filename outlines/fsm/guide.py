import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, Union

import interegular
import torch
from lark import Lark
from collections import defaultdict

from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import (
    create_fsm_index_tokenizer,
    make_byte_level_fsm,
    make_deterministic_fsm,
)
import dataclasses
from collections import defaultdict


if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer

@dataclasses.dataclass
class JumpEdge:
    symbol: str = None
    symbol_next_state: int = None
    byte: int = None
    byte_next_state: int = None



@dataclass(frozen=True)
class Write:
    """Write instruction.

    Attributes
    ----------
    tokens
        The sequence of tokens to be added to the current sequence by the
        generation process.

    """

    tokens: torch.Tensor


@dataclass(frozen=True)
class Generate:
    """Generate instruction

    Attributes
    ----------
    tokens
        The tokens that lead to a valid completion if generated.  A value
        of ``None`` indicates that all tokens are allowed.
    """

    tokens: torch.Tensor


Instruction = Union[Write, Generate]


class Guide(Protocol):
    """Base definition of a generation guide.

    A generation guide defines the behavior of a finite-state machine that guides
    a text generation procedure. Unlike the DFAs built from regular expressions
    guides can also emit a `Write` instructions which tells the model that it can
    append a sequence of tokens (or token word) instead of generating it.

    """

    def get_next_instruction(self, state: int) -> Instruction:
        ...

    def get_next_state(self, state: int, token_id: int) -> int:
        ...

    def is_final_state(self, state: int) -> bool:
        ...

    def copy(self) -> "Guide":
        ...


class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""

    final_state = 1
    start_state = 0

    def __init__(self, tokenizer: "Tokenizer"):
        """Initialize the generation guide.

        model
            The logit generator used to generate the next token.

        """
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary.values()

    def get_next_instruction(self, state: int) -> Instruction:
        if self.is_final_state(state):
            return Write([self.eos_token_id])
        return Generate(None)

    def get_next_state(self, state: int, token_id: int) -> int:
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state

        return self.start_state

    def is_final_state(self, state: int):
        return state == self.final_state

    def copy(self):
        return self


@cache()
def create_states_mapping(
        regex_string: str, tokenizer: "Tokenizer"
) -> Tuple[dict, set, set, dict]:
    """Create the variables related to the mapping between states and tokens
    The parameters of the function are used for caching purpose
    """
    regex_pattern = interegular.parse_pattern(regex_string)
    byte_fsm = make_byte_level_fsm(regex_pattern.to_fsm().reduce(), keep_utf8=True)
    regex_fsm, _ = make_deterministic_fsm(byte_fsm)
    states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
        regex_fsm, tokenizer
    )

    # We make sure that it is possible to generate strings in the language
    # of the regular expression with the tokens present in the model's
    # vocabulary.
    if not any(
            regex_fsm.finals.intersection(v.values()) for v in states_to_token_maps.values()
    ):
        raise ValueError(
            "The vocabulary does not allow us to build a sequence that matches the input regex"
        )

    try:  # FIXME(liuquan) regex_fsm.fsm_info.alphabet_symbol_mapping  can not (for)
        state_to_leap_forward = self._init_state_to_leap_forward(regex_fsm)
        logging.info(f"{regex_string} state_to_leap_forward is created")
    except:
        logging.warning(f"{regex_string} state_to_leap_forward failed to create")
        state_to_leap_forward = {}

    return states_to_token_maps, empty_token_ids, regex_fsm.finals, state_to_leap_forward


class RegexGuide(Guide):
    """Guide to generate text in the language of a regular expression."""

    initial_state = 0

    def __init__(self, regex_string: str, tokenizer: "Tokenizer", mode="comp"):
        (
            self.states_to_token_maps,
            self.empty_token_ids,
            fsm_finals,
            self.state_to_leap_forward,
        ) = create_states_mapping(regex_string, tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}
        self.mode = mode

        if self.mode == "comp":
            self._compress_fsm()
        else:
            self._cache_state_to_token_tensor()

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the guide.

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """
        if self.mode == "comp":
            next_tensor_ids = self.states_to_token_maps.get(state)
            if next_tensor_ids is None:
                next_tokens_mask = None
            else:
                next_tokens_mask = torch.cat([self.token_tensor_maps.get(tensor_id) for tensor_id in next_tensor_ids],
                                             dim=0)
        else:
            next_tokens_mask = self.states_to_token_mask.get(state)

        if next_tokens_mask is None:
            return Write(torch.tensor([self.eos_token_id]))

        return Generate(next_tokens_mask)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        We use the index to determine to which state the guide should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the guide.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the guide.

        """
        if token_id == self.eos_token_id or state not in self.states_to_token_maps:
            return -1

        if self.mode == "comp":
            last_token_to_end_state = self.states_to_token_maps[state]
            next_state = None

            for i in last_token_to_end_state:
                if token_id in self.token_tensor_maps[str(i)]:
                    tensor_id = str(i)
                    next_state = self.path_maps[tensor_id].get(state)
                    break

            if next_state is None:
                next_state = self.final_state
            return next_state

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            return self.final_state

        return next_state

    @classmethod
    def from_interegular_fsm(
            cls, interegular_fsm: interegular.fsm.FSM, tokenizer: "Tokenizer"
    ):
        from_interegular_instance = cls.__new__(cls)

        def create_states_mapping_from_interegular_fsm(
                fsm: interegular.fsm.FSM,
        ) -> Tuple[dict, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            byte_fsm = make_byte_level_fsm(fsm.reduce(), keep_utf8=True)
            regex_fsm, _ = make_deterministic_fsm(byte_fsm)
            states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
                regex_fsm, tokenizer
            )

            # We make sure that it is possible to generate strings in the language
            # of the regular expression with the tokens present in the model's
            # vocabulary.
            if not any(
                    regex_fsm.finals.intersection(v.values())
                    for v in states_to_token_maps.values()
            ):
                raise ValueError(
                    "The vocabulary does not allow us to build a sequence that matches the input regex"
                )

            return states_to_token_maps, empty_token_ids

        (
            from_interegular_instance.states_to_token_maps,
            from_interegular_instance.empty_token_ids,
        ) = create_states_mapping_from_interegular_fsm(interegular_fsm)
        from_interegular_instance.eos_token_id = tokenizer.eos_token_id
        from_interegular_instance._cache_state_to_token_tensor()
        return from_interegular_instance

    def _cache_state_to_token_tensor(self):
        """
        cache state -> token int tensor
        this increases performance of mask construction substantially
        """
        self.states_to_token_mask = {
            state: torch.tensor(list(next_tokens_to_end_states.keys()))
            for state, next_tokens_to_end_states in self.states_to_token_maps.items()
        }

    def _compress_fsm(self):
        # same next state tokens as tokens_set eg:{"0->1":[1,2,3,4],"0->2":[56,67,78,65]}
        # form state0 to state1 have  [1,2,3,4] tokens_set.
        state_state_to_edge = defaultdict(list)
        for in_state, token_to_states in self.states_to_token_maps.items():
            for token, out_state in token_to_states.items():
                state_state_to_edge[str(in_state) + "->" + str(out_state)].append(token)

        # same tokens_set to state transfer eg:{(1,2,3,4):,["3->6","0->2"]} (1,2,3,4)
        # tokens_set have [form state0 to state2] and  [form state3 to state6] .
        edge_to_state_state = {}
        for k, v in state_state_to_edge.items():
            sorted_token = tuple(sorted(v))
            if sorted_token in edge_to_state_state.keys():
                edge_to_state_state[sorted_token].append(k)
            else:
                edge_to_state_state[sorted_token] = [k]

        # token_tensor_maps:{tokens_set_id: tensor(tokens_set)}, each tokens_set corresponds to an id.
        # path_maps: {tokens_set_id: {statex:statey}}，tokens_set_id corresponds status change
        # states_to_token_maps: {state:[tokens_set_ids]}, each state contains tokens_sets.
        self.token_tensor_maps = defaultdict()
        self.path_maps = defaultdict(dict)
        for i, (k, v) in enumerate(edge_to_state_state.items()):
            self.token_tensor_maps[str(i)] = torch.tensor(list(k))
            for path in v:
                in_state, out_state = path.split("->")[0], path.split("->")[1]
                self.path_maps[str(i)][int(in_state)] = int(out_state)

                if isinstance(self.states_to_token_maps[int(in_state)], list):
                    self.states_to_token_maps[int(in_state)].append(str(i))
                else:
                    self.states_to_token_maps[int(in_state)] = [str(i)]

    def _init_state_to_leap_forward(self, regex_fsm):
        fsm_info = regex_fsm.fsm_info

        symbol_to_id = fsm_info.alphabet_symbol_mapping
        id_to_symbol = {}
        for symbol, id_ in symbol_to_id.items():
            id_to_symbol.setdefault(id_, []).append(symbol)

        transitions = fsm_info.transitions

        outgoings_ct = defaultdict(int)
        # NOTE(lsyin): Final states can lead to terminate, so they have one outgoing edge naturally
        for s in fsm_info.finals:
            outgoings_ct[s] = 1

        state_to_leap_forward = {}
        for (state, id_), next_state in transitions.items():
            if id_ == fsm_info.alphabet_anything_value:
                # Arbitrarily symbol cannot be recognized as jump forward
                continue

            symbols = id_to_symbol[id_]
            for c in symbols:
                if len(c) > 1:
                    # Skip byte level transitions like c = "5E"
                    continue

                outgoings_ct[state] += 1
                if outgoings_ct[state] > 1:
                    if state in state_to_leap_forward:
                        del state_to_leap_forward[state]
                    break

                state_to_leap_forward[state] = JumpEdge(
                    symbol=c,
                    symbol_next_state=next_state,
                )

        # Process the byte level jump forward
        outgoings_ct = defaultdict(int)
        for s in fsm_info.finals:
            outgoings_ct[s] = 1

        for (state, id_), next_state in transitions.items():
            if id_ == fsm_info.alphabet_anything_value:
                continue
            symbols = id_to_symbol[id_]
            for c in symbols:
                byte_ = None
                if len(c) == 1 and ord(c) < 0x80:
                    # ASCII character
                    byte_ = ord(c)
                elif len(c) > 1:
                    # FIXME: This logic is due to the leading \x00
                    # https://github.com/outlines-dev/outlines/pull/930
                    byte_ = int(symbols[0][1:], 16)

                if byte_ is not None:
                    outgoings_ct[state] += 1
                    if outgoings_ct[state] > 1:
                        if state in state_to_leap_forward:
                            del state_to_leap_forward[state]
                        break
                    e = state_to_leap_forward.get(state, JumpEdge())
                    e.byte = byte_
                    e.byte_next_state = next_state
                    state_to_leap_forward[state] = e

        return state_to_leap_forward

    def is_final_state(self, state: int) -> bool:
        """Determine whether the current state of the guide is a final state."""
        return state in self.final_states

    def copy(self):
        return self


class CFGGuide(Guide):
    """Guide to generate text that is in the language of a context-free grammar."""

    def __init__(self, cfg_string: str, tokenizer):
        self.cfg_string = cfg_string
        self.tokenizer = tokenizer

        self.parser = Lark(
            cfg_string,
            parser="lalr",
            lexer="contextual",
            propagate_positions=False,
            maybe_placeholders=False,
            regex=True,
            import_paths=[grammars.GRAMMAR_PATH],
        )
        self.terminal_regexps = dict()
        for terminal in self.parser.terminals:
            if terminal.pattern is not None:
                self.terminal_regexps[terminal.name] = terminal.pattern.to_regexp()
        self.terminal_regexps["$END"] = tokenizer.eos_token

        self.generation = ""
        self.reset_state = False
        self.allow_eos = False
        self.regex_fsm: RegexGuide

        self.check_last = False
        self.proposal_last: List[int] = []
        self.regex_fsm_last: RegexGuide

        self.start_state = 0
        self.final_state = -1

    def get_next_instruction(self, state: int) -> Instruction:
        """Generate an instruction for the next step.

        Upon initialization, the CFG incremental parser is used to determine the
        first regex and construct the first FSM to generate the first terminal.

        This FSM is used for proposals until either:

        - The FSM is exhausted, and its only remaining option is the EOS token,
          in which case we feed the generated terminal to the
          CFG incremental parser and allow it to propose the next regex
          corresponding to the next set of valid terminals.
        - The current FSM can be exhausted, but the EOS token is not the only
          remaining option. In this case we allow proposal of current terminal
          extensions, store the current FSM and its state, then also use the CFG
          parser to propose a new regex corresponding to terminating the current
          terminal and starting the next one. The model can then sample from
          either of these sets to determine whether to extend the current
          terminal or terminate it and start the next one.

        The CFG incremental parser is allowed to propose the EOS token from any accepting state,
        and once it is generated, the FSM will continue to always generate the EOS token.

        Parameters
        ----------
        state
            The current state of the FSM.

        Returns
        -------
        A list that contains the tokens to mask.

        """
        if self.is_final_state(state):
            return Write([self.tokenizer.eos_token_id])

        proposal: List[int] = []
        if self.generation != "":
            if self.check_last:
                proposer = self.regex_fsm_last
            else:
                proposer = self.regex_fsm

            instruction = proposer.get_next_instruction(state)

            assert instruction.tokens is not None

            if isinstance(instruction, Write):
                proposal += instruction.tokens
            else:
                proposal += instruction.tokens

            if self.tokenizer.eos_token_id not in proposal:
                return Generate(proposal)

            self.check_last = False
            proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
            if len(proposal) > 0:
                self.check_last = True
                self.proposal_last = proposal.copy()
                self.regex_fsm_last = proposer

        interactive = self.parser.parse_interactive(self.generation)
        interactive.exhaust_lexer()

        options = {self.terminal_regexps[x] for x in interactive.accepts()}
        # add %ignore terminals
        options |= {self.terminal_regexps[x] for x in self.parser.lexer_conf.ignore}

        if self.terminal_regexps["$END"] in options:
            options.remove(self.terminal_regexps["$END"])
            if len(options) == 0:
                return Write([self.tokenizer.eos_token_id])
            self.allow_eos = True
            options.add("")
            assert len(options) > 1

        regex_string = r"(" + r"|".join([r"(" + x + r")" for x in options]) + r")"
        self.regex_fsm = RegexGuide(regex_string, self.tokenizer)
        self.reset_state = True

        instruction = self.regex_fsm.get_next_instruction(self.start_state)

        assert instruction.tokens is not None

        if isinstance(instruction, Write):
            proposal += instruction.tokens
        else:
            proposal += instruction.tokens

        if self.allow_eos:
            self.allow_eos = False
        else:
            proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
            assert len(proposal) > 0

        return Generate(proposal)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        Transitions the underlying regex FSM to its next state.
        If at max tokens or EOS token, transition permanently to the final state.
        Update stored partial generations for subsequent incremental parsing.

        Parameters
        ----------
        state
            The current state of the FSM.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the FSM.
        """

        # We need to return the final state when in the final state because we
        # then generate EOS tokens instead of stopping the generation.
        if token_id == self.tokenizer.eos_token_id or state == self.final_state:
            return self.final_state

        self.generation += self.tokenizer.decode([token_id])[0]

        if self.check_last:
            if token_id in self.proposal_last:
                return self.regex_fsm_last.get_next_state(state, token_id)
            self.check_last = False

        if self.reset_state:
            self.reset_state = False
            state = self.start_state

        return self.regex_fsm.get_next_state(state, token_id)

    def is_final_state(self, state: int) -> bool:
        return state == self.final_state

    def copy(self) -> "CFGGuide":
        """Create a copy of the FSM."""
        return CFGGuide(self.cfg_string, self.tokenizer)
