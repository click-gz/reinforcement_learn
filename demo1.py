import numpy as np
import pickle

ROWS = 4
COLS = 4
WON = 3


class ENV:
    def __init__(self, rows, cols, won):
        self.rows = rows
        self.cols = cols
        self.won = won
        self.data = np.zeros((self.rows, self.cols))
        self.winner = None
        self.hash_val = None
        self.done = None

    def check_five_in_a_row(self, player, row, col, direction):
        """
        检查某个位置在给定方向上是否连成五子。
        Args:
            board: 二维列表，表示棋盘。
            player: 当前玩家，1 表示黑子，2 表示白子。
            row: 当前位置的行索引。
            col: 当前位置的列索引。
            direction: (dx, dy)，表示检查方向的增量。

        Returns:
            True：如果在给定方向上连成五子。
            False：否则。
        """
        for i in range(5):
            r = row + i * direction[0]
            c = col + i * direction[1]
            if r < 0 or r >= len(self.data) or c < 0 or c >= len(self.data[0]) or self.data[r][c] != player:
                return False
        return True

    def Done(self):
        """
            判断五子棋游戏是否结束。
            Args:
                self.data: 二维列表，表示棋盘。
                player: 当前玩家，1 表示黑子，2 表示白子。

            Returns:
                True：如果游戏结束（某一方连成五子）。
                False：否则。
            """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 四个方向：水平、垂直、对角线、反对角线
        for i in [-1, 1]:
            for row in range(len(self.data)):
                for col in range(len(self.data[0])):
                    for direction in directions:
                        if self.check_five_in_a_row(i, row, col, direction):
                            self.winner = i
                            self.done = True
                            return self.done

        data_sum = np.sum(np.abs(self.data))
        if data_sum == self.rows * self.cols:
            self.winner = 0
            self.done = True
            return self.done

        self.done = False
        return self.done

    def next_state(self, i, j, symbol):
        new_state = ENV(self.rows, self.cols, self.won)
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # print the board
    def print(self):
        for i in range(0, self.rows):
            print('-------------')
            out = '| '
            for j in range(0, self.cols):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')
        # compute the hash value for one state, it's unique

    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in self.data.reshape(self.rows * self.cols):
                if i == -1:
                    i = 2
                self.hash_val = self.hash_val * self.won + i
        return int(self.hash_val)


def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(0, ROWS):
        for j in range(0, COLS):
            print("len: ", len(all_states))
            if current_state.data[i][j] == 0:
                newState = current_state.next_state(i, j, current_symbol)
                newHash = newState.hash()
                if newHash not in all_states.keys():
                    isEnd = newState.Done()
                    all_states[newHash] = (newState, isEnd)
                    if not isEnd:
                        get_all_states_impl(newState, -current_symbol, all_states)


def get_all_states():
    current_symbol = 1
    current_state = ENV(ROWS, COLS, WON)
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.Done())
    print("start;;;;;")
    get_all_states_impl(current_state, current_symbol, all_states)
    print("over;;;;;;")
    return all_states


# all possible board configurations
print("-------------loading states-------------")
all_states = get_all_states()
print("-------------loading over---------------")


class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self, printf=False):
        alternator = self.alternate()
        self.reset()
        current_state = ENV(ROWS, COLS, WON)
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        while True:
            player = next(alternator)
            if printf:
                current_state.print()
            [i, j, symbol] = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if is_end:
                if printf:
                    current_state.print()
                return current_state.winner


# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states.keys():
            (state, is_end) = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    # update value estimation
    def backup(self):
        # for debug
        # print('player trajectory')
        # for state in self.states:
        #     state.print()

        self.states = [state.hash() for state in self.states]

        for i in reversed(range(len(self.states) - 1)):
            state = self.states[i]
            td_error = self.greedy[i] * (self.estimations[self.states[i + 1]] - self.estimations[state])
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(ROWS):
            for j in range(COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())

        # 截断
        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash], pos))
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)

        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)


# human interface
# input a number to put a chessman
# | q | w | e | r
# | a | s | d | f
# | z | x | c | v
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']
        self.state = None
        return

    def reset(self):
        return

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol
        return

    def backup(self, _):
        return

    def act(self):
        self.state.print()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // int(COLS)
        j = data % COLS
        return (i, j, self.symbol)


def train(epochs):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(printf=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        print('Epoch %d, player 1 win %.02f, player 2 win %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()
        judger.reset()
    player1.save_policy()
    player2.save_policy()


def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for i in range(0, turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")


if __name__ == '__main__':
    print("train_start-------------")
    train(int(1e5))
    compete(int(1e3))
    play()
