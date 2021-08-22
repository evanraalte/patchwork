from __future__ import annotations
import yaml
from collections import namedtuple
from enum import Enum, auto
from typing import Dict, List, Set
from dataclasses import dataclass, field, InitVar
import copy

INITIAL_BUTTONS = 5


class InsufficientButtonsException(Exception):
    """THrown if a patch doesn't fit on the quiltboard"""


@dataclass
class CentralTimeBoard:
    buttons_7x7_available: bool = True
    final_position: int = 53
    button_positions: Set = field(default_factory=lambda: set({5, 11, 17, 23, 29, 35, 41, 47, 53}))
    player_positions: Dict = field(default_factory=lambda: {0: 0, 1: 0})
    leather_positions: Set = field(default_factory=lambda: set({26, 32, 38, 44, 50}))


Orientation = namedtuple("Orientation", "rotation mirror")
Coordinate = namedtuple("Coordinate", "x y")


class Rotation(Enum):
    CW0 = auto()
    CW90 = auto()
    CW180 = auto()
    CW270 = auto()


class IllegalRotationException(Exception):
    """Thrown if a patch doesn't fit on the quiltboard"""

    def __init__(self, rotation):
        self.rotation = rotation
        super().__init__("This rotation is not supported")

    def __str__(self):
        return f'{self.rotation=}'


class Shape:
    base = Orientation(rotation=Rotation.CW0, mirror=False)

    def __eq__(self, o: Shape) -> bool:
        str(self) == str(o)

    def __str__(self):
        buf = ""
        for coords in self.coordinates.values():
            x_max = max(map(lambda c: c.x, coords))
            y_max = max(map(lambda c: c.y, coords))
            for y in range(0, y_max + 1):
                for x in range(0, x_max + 1):
                    buf += '# ' if (x, y) in coords else '  '
                buf += '\n'
            buf += '\n'
        return buf

    def __init__(self, shape_list: List[List[int]]):
        self.coordinates: Set[Coordinate] = {}
        rotations = set([Rotation.CW0])
        mirror_states = set([False])

        self.coordinates[self.base] = {
            Coordinate(c, r) for (r, line) in enumerate(shape_list) for (c, e) in enumerate(line) if e == 1
        }

        if self.coordinates[self.base] != self.mirror(self.coordinates[self.base], True):  # can be mirrored
            mirror_states.add(True)

        for r in Rotation:
            if self.coordinates[self.base] != self.rotate_cw(self.coordinates[self.base], r):  # can be rotated 90 cw
                rotations.add(r)

        for c in rotations:
            for m in mirror_states:
                if (c, m) == self.base:
                    continue
                t_shape = self.rotate_cw(self.mirror(self.coordinates[self.base], m), c)
                if t_shape in self.coordinates.values():
                    continue
                self.coordinates[Orientation(c, m)] = t_shape

    @classmethod
    def offset_to_zero(cls, coordinates: Set[Coordinate]):  # Align x_min and y_min to 0,0
        (x_min, _, y_min, _) = cls.get_bounds(coordinates)
        return {Coordinate(x - x_min, y - y_min) for (x, y) in coordinates}

    @staticmethod
    def offset(position: Coordinate, coordinates: Set[Coordinate]):  # Align x_min and y_min to coordinates
        offset_x, offset_y = position
        return {Coordinate(x + offset_x, y + offset_y) for (x, y) in coordinates}

    @staticmethod
    def get_bounds(coordinates: Set[Coordinate]):
        x, y = tuple(zip(*coordinates))
        return (min(x), max(x), min(y), max(y))

    @classmethod
    def mirror(cls, coordinates: Set[Coordinate], mirror: bool):
        if mirror:
            return cls.offset_to_zero({Coordinate(x, -y) for (x, y) in coordinates})
        else:
            return coordinates

    @classmethod
    def rotate_cw(cls, coordinates: Set[Coordinate], rotation: Rotation):
        if rotation == Rotation.CW0:
            rotated_shape = coordinates
        elif rotation == Rotation.CW90:
            rotated_shape = {Coordinate(y, -x) for (x, y) in coordinates}
        elif rotation == Rotation.CW180:
            rotated_shape = {Coordinate(-x, -y) for (x, y) in coordinates}
        elif rotation == Rotation.CW270:
            rotated_shape = {Coordinate(-y, x) for (x, y) in coordinates}
        else:
            raise IllegalRotationException(rotation)
        return cls.offset_to_zero(rotated_shape)


@dataclass
class Patch:
    time_penalty: int = 0
    cost: int = 0
    income: int = 0
    sl: InitVar[List[List[int]]] = None
    shape: Shape = field(init=False)

    def __eq__(self, o: Patch) -> bool:
        return (
            self.time_penalty == o.time_penalty
            and self.cost == o.cost
            and self.income == o.income
            and self.shape == o.shape
        )

    def __post_init__(self, sl):
        self.shape = Shape(sl)


class PlacementException(Exception):
    """Thrown if a patch doesn't fit on the quiltboard"""

    def __init__(self, patch, location, orientation):
        self.patch = patch
        self.location = location
        self.orientation = orientation
        super().__init__("Patch does not fit on QuiltBoard")

    def __str__(self):
        return f'{self.patch=}, {self.location=}, {self.orientation=}'


class QuiltBoard:
    def __init__(self):
        self.empty_tiles = {(x, y) for y in range(0, 9) for x in range(0, 9)}
        self.button_cnt = 0

    def __eq__(self, o: QuiltBoard) -> bool:
        return self.empty_tiles == o.empty_tiles and self.button_cnt == o.button_cnt

    def __str__(self):
        buf = ""
        for y in range(0, 9):
            for x in range(0, 9):
                buf += '# ' if (x, y) not in self.empty_tiles else '. '
            buf += '\n'
        return buf

    def add_patch(self, patch: Patch, location: Coordinate, orientation: Orientation):
        if not self.fits_patch(patch, location, orientation):
            raise PlacementException(patch, location, orientation)
        self.button_cnt += patch.income
        self.empty_tiles -= patch.shape.offset(location, patch.shape.coordinates[orientation])

    def fits_patch(self, patch: Patch, location: Coordinate, orientation: Orientation) -> bool:
        coordinates = patch.shape.offset(location, patch.shape.coordinates[orientation])
        return all(c in self.empty_tiles for c in coordinates)

    @staticmethod
    def _generate_7x7(x_offset, y_offset):
        return {(x, y) for y in range(x_offset, x_offset + 7) for x in range(y_offset, y_offset + 7)}

    def has7x7(self):
        for d_col in range(0, 3):
            for d_row in range(0, 3):
                grid = self._generate_7x7(d_col, d_row)
                if grid - self.empty_tiles == grid:
                    return True
        return False

    def get_empty_tiles(self):
        return len(self.empty_tiles)


class PatchCircle:

    leather_patch = Patch(time_penalty=0, cost=0, income=0, sl=[[1]])

    def __init__(self) -> None:
        self.neutral_token = 0
        # read patches from yaml
        with open("patches.yaml") as f:
            patches = yaml.safe_load(f)
        self.patches = [Patch(**patch) for patch in patches]
        # randomize order
        pass

    def __eq__(self, o: PatchCircle) -> bool:
        return self.neutral_token == o.neutral_token and self.patches == o.patches

    def get(self, relative_patch_position, pop=False) -> Patch:
        idx = (self.neutral_token + relative_patch_position) % len(self.patches)
        patch = self.patches[idx]
        if pop:
            self.patches.pop(idx)
        return patch


@dataclass
class Player:
    id: int = None
    opponent_id: int = field(init=False)
    central_time_board: CentralTimeBoard = None
    patch_circle: PatchCircle = None
    buttons: int = field(init=False, default=INITIAL_BUTTONS)
    position: int = field(init=False, default=0)
    quilt_board: QuiltBoard = field(init=False)
    final_move_played: bool = field(init=False, default=False)
    move_finished: bool = False

    def __eq__(self, o) -> bool:
        return self.buttons == o.buttons and self.position == o.position and self.quilt_board == o.quilt_board

    def __post_init__(self) -> None:
        self.opponent_id = 1 - self.id
        self.quilt_board = QuiltBoard()

    def __str__(self) -> str:
        return f"{self.id=} {self.buttons=} {self.position=}"

    def increment_position(self, steps, patch_position=None) -> None:
        buttons = self.buttons
        final_move_played = False
        move_finished = False

        if self.position + steps >= self.central_time_board.final_position:
            steps = self.central_time_board.final_position - self.position
            final_move_played = True

        if self.position + steps > self.central_time_board.player_positions[self.opponent_id]:
            move_finished = True

        for i in range(self.position + 1, self.position + steps + 1):  # check if we walked over a button
            if i in self.central_time_board.button_positions:
                buttons = self.buttons + self.quilt_board.button_cnt
            if i in self.central_time_board.leather_positions:
                assert patch_position is not None
                self.quilt_board.add_patch(self.patch_circle.leather_patch, patch_position)  # place it on the board
                self.central_time_board.leather_positions.remove(i)  # take the patch

        self.buttons = buttons
        self.final_move_played = final_move_played
        self.move_finished = move_finished
        self.position += steps
        self.central_time_board.player_positions[self.id] = self.position

    def add_buttons(self, num) -> None:
        self.buttons = self.buttons + num

    def rm_buttons(self, num) -> None:
        self.buttons = self.buttons - num

    def buy_patch(self, relative_patch_position: int) -> Patch:
        patch = self.patch_circle.get(relative_patch_position, pop=False)
        if patch.cost > self.buttons:
            raise InsufficientButtonsException()
        else:
            self.rm_buttons(patch.cost)
            return self.patch_circle.get(relative_patch_position, pop=True)

    def play_advance_and_receive_buttons(self, patch_position) -> bool:
        position_other_player = self.central_time_board.player_positions[self.opponent_id]
        diff = position_other_player - self.position
        assert diff >= 0

        try:
            self.increment_position(diff + 1, patch_position)
            self.add_buttons(diff + 1)
            return True
        except PlacementException:
            return False

    # validity of move to be determined in arbiter
    def play_take_and_place_a_patch(
        self, relative_patch_position: int, location: Coordinate, orientation: Orientation
    ) -> bool:
        # Check that it is actually our turn
        position_other_player = self.central_time_board.player_positions[self.opponent_id]
        diff = position_other_player - self.position
        assert diff >= 0

        try:
            patch = self.buy_patch(relative_patch_position)
            self.quilt_board.add_patch(patch, location, orientation)
            self.increment_position(patch.time_penalty)
            return True
        except (PlacementException, InsufficientButtonsException):
            return False

    def get_points(self) -> int:
        return self.buttons - 2 * self.quilt_board.get_empty_tiles()


class Arbiter:
    # Needs to know player positions
    def __eq__(self, o: Arbiter) -> bool:
        return (
            self.central_time_board == o.central_time_board
            and self.players == o.players
            and self.patch_circle == o.patch_circle
            and self.player_id == o.player_id
        )

    def switch_player(self, player_id=None) -> Player:
        if player_id is not None:
            self.player_id = player_id
        else:
            self.player_id = 1 - self.player_id
        return self.players[self.player_id]

    def __init__(self) -> None:
        self.central_time_board = CentralTimeBoard()
        self.patch_circle = PatchCircle()
        self.players = {}
        self.players[0] = Player(id=0, central_time_board=self.central_time_board, patch_circle=self.patch_circle)
        self.players[1] = Player(id=1, central_time_board=self.central_time_board, patch_circle=self.patch_circle)

    def copy(self) -> Arbiter:
        return copy.deepcopy(self)

    def run(self) -> None:
        self.p = self.switch_player(0)
        game_complete = False

        while not game_complete:
            # Check if the game is done
            if all(map(lambda _p: _p.final_move_played, self.players.values())):
                print(f"Game complete, p0: {self.players[0].get_points()}, p1: {self.players[1].get_points()}")
                print(f"p0:\n{str(self.players[0].quilt_board)}")
                print(f"p1:\n{str(self.players[1].quilt_board)}")
                game_complete = True
                break

            # Otherwise, calculate all next moves
            next_moves: List[Arbiter] = []
            for tile in self.p.quilt_board.empty_tiles:
                # 1
                _self = self.copy()
                valid_move = _self.p.play_advance_and_receive_buttons(tile)
                if _self.p.move_finished:
                    _self.p = _self.switch_player()
                if valid_move and (_self not in next_moves):
                    # print(f'play_advance_and_receive_buttons {tile=}')
                    print(f'{_self.players[0].quilt_board}')
                    next_moves.append(_self)  # add this state as a valid move

                # 2
                # for i in range(0, 1):
                #     _self = self.copy()
                #     orientations = _self.patch_circle.get(i, pop=False).shape.coordinates.keys()
                #     for orientation in orientations:
                #         valid_move = _self.p.play_take_and_place_a_patch(i, tile, orientation)
                #         if _self.p.move_finished:
                #             _self.p = _self.switch_player()
                #         if valid_move and (_self not in next_moves):
                #             next_moves.append(_self)
                #             print(f'{_self.players[0].quilt_board}')
            print(f"possible moves in this state: {len(next_moves)}")
            return

        pass


def main():
    arbiter = Arbiter()
    arbiter.run()


main()
