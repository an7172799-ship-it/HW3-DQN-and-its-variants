from GridBoard import GridBoard, randPair, addTuple


class Gridworld:
    """
    4x4 GridWorld from DRL in Action (Chapter 3).

    Modes:
        static  - fixed layout: Player(0,3), Goal(0,0), Pit(0,1), Wall(1,1)
        player  - only Player position randomized; others fixed
        random  - all pieces randomized

    State: game.board.render_np().reshape(1, 64)
        → 4 channels × 4×4 grid, each channel is one piece's position (one-hot)
        → flattened to 64 dims

    Actions via makeMove(): 'u' up, 'd' down, 'l' left, 'r' right
    Reward via reward():    +1 goal, -1 pit, 0 otherwise
    """
    def __init__(self, size=4, mode='static'):
        self.board = GridBoard(size=max(size, 4))
        self.board.addPiece('Player', 'P', (0, 0))
        self.board.addPiece('Goal',   '+', (1, 0))
        self.board.addPiece('Pit',    '-', (2, 0))
        self.board.addPiece('Wall',   'W', (3, 0))

        if mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

    def initGridStatic(self):
        self.board.components['Player'].pos = (0, 3)
        self.board.components['Goal'].pos   = (0, 0)
        self.board.components['Pit'].pos    = (0, 1)
        self.board.components['Wall'].pos   = (1, 1)

    def validateBoard(self):
        player = self.board.components['Player'].pos
        goal   = self.board.components['Goal'].pos
        wall   = self.board.components['Wall'].pos
        pit    = self.board.components['Pit'].pos
        all_positions = [player, goal, wall, pit]
        if len(all_positions) > len(set(all_positions)):
            return False
        corners = [(0, 0), (0, self.board.size-1),
                   (self.board.size-1, 0), (self.board.size-1, self.board.size-1)]
        if player in corners or goal in corners:
            val_pl = [self.validateMove('Player', d) for d in [(0,1),(1,0),(-1,0),(0,-1)]]
            val_go = [self.validateMove('Goal',   d) for d in [(0,1),(1,0),(-1,0),(0,-1)]]
            if 0 not in val_pl or 0 not in val_go:
                return False
        return True

    def initGridPlayer(self):
        self.initGridStatic()
        self.board.components['Player'].pos = randPair(0, self.board.size)
        if not self.validateBoard():
            self.initGridPlayer()

    def initGridRand(self):
        self.board.components['Player'].pos = randPair(0, self.board.size)
        self.board.components['Goal'].pos   = randPair(0, self.board.size)
        self.board.components['Pit'].pos    = randPair(0, self.board.size)
        self.board.components['Wall'].pos   = randPair(0, self.board.size)
        if not self.validateBoard():
            self.initGridRand()

    def validateMove(self, piece, addpos=(0, 0)):
        """Returns 0=valid, 1=invalid(wall/boundary), 2=pit."""
        pit  = self.board.components['Pit'].pos
        wall = self.board.components['Wall'].pos
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        if new_pos == wall:
            return 1
        if max(new_pos) > (self.board.size - 1) or min(new_pos) < 0:
            return 1
        if new_pos == pit:
            return 2
        return 0

    def makeMove(self, action):
        moves = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1)}
        addpos = moves[action]
        if self.validateMove('Player', addpos) in [0, 2]:
            new_pos = addTuple(self.board.components['Player'].pos, addpos)
            self.board.movePiece('Player', new_pos)

    def reward(self):
        player = self.board.components['Player'].pos
        if player == self.board.components['Pit'].pos:
            return -1
        if player == self.board.components['Goal'].pos:
            return 1
        return 0

    def display(self):
        return self.board.render()
