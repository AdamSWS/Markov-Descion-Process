class GridworldMDP:
    def __init__(self, size, walls, terminal_states, reward, transition_probabilities, discount_rate):
        self.size = size
        self.walls = walls
        self.terminal_states = terminal_states
        self.reward = reward
        self.transition_probabilities = transition_probabilities
        self.discount_rate = discount_rate

    @property
    def states(self):
        all_states = [
            (col, row)
            for row in range(1, self.size[1] + 1)
            for col in range(1, self.size[0] + 1)
            if (col, row) not in self.walls
        ]
        return all_states

    def S(self):
        return [(col, row) for col in range(1, self.size[0] + 1) for row in range(1, self.size[1] + 1)]

    def A(self, state):
        if state in self.walls or state in self.terminal_states:
            return []
        return [(0, -1), (-1, 0), (0, 1), (1, 0)]

    def P(self, state, action):
        next_states = {}
        for prob_idx, prob in enumerate(self.transition_probabilities):
            next_col, next_row = state[0] + self.A(state)[(self.A(state).index(action) + prob_idx) % 4][0], \
                                 state[1] + self.A(state)[(self.A(state).index(action) + prob_idx) % 4][1]
            if not (1 <= next_col <= self.size[0] and 1 <= next_row <= self.size[1]):
                next_col, next_row = state
            elif (next_col, next_row) in self.walls:
                next_col, next_row = state
            next_states[(next_col, next_row)] = prob
        return next_states

    def R(self, state, action, next_state):
        if next_state in self.terminal_states:
            return self.terminal_states[next_state]
        return self.reward

    def is_terminal(self, state):
        return state in self.terminal_states

def read_input(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    config = {}
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        key, value = line.strip().split(':', 1)
        config[key.strip()] = value.strip()

    return config

def parse_input(config):
    size = tuple(map(int, config['size'].split()))
    walls = [tuple(map(int, wall.split())) for wall in config['walls'].split(',')]
    walls = [(wall[0], wall[1]) for wall in walls]
    terminal_states = [tuple(map(int, state.split())) for state in config['terminal_states'].split(',')]
    terminal_states = {(state[0], state[1]): state[2] for state in terminal_states}
    reward = float(config['reward'])
    transition_probabilities = list(map(float, config['transition_probabilities'].split()))
    discount_rate = float(config['discount_rate'])
    epsilon = float(config['epsilon'])

    return size, walls, terminal_states, reward, transition_probabilities, discount_rate, epsilon

def Q_value(mdp, state, action, U):
    sum_rewards = 0

    # Define the transition probabilities
    transition_probs = [0.8, 0.1, 0, 0.1]

    for prob_idx, prob in enumerate(transition_probs):
        next_col, next_row = state[0] + mdp.A(state)[(mdp.A(state).index(action) + prob_idx) % 4][0], \
                             state[1] + mdp.A(state)[(mdp.A(state).index(action) + prob_idx) % 4][1]
        if not (1 <= next_col <= mdp.size[0] and 1 <= next_row <= mdp.size[1]):
            next_col, next_row = state
        elif (next_col, next_row) in mdp.walls:
            next_col, next_row = state
        next_states = (next_col, next_row)

        reward = mdp.R(state, action, next_states)
        sum_rewards += prob * (reward + mdp.discount_rate * U[next_states])
    return sum_rewards

def value_iteration_v2(mdp, epsilon):
    U = {state: 0 for state in mdp.states}
    U_prime = U.copy()
    delta = 0
    iteration = 0
    print("################ VALUE ITERATION ###########################\n")

    while True:
        U = U_prime.copy()
        delta = 0

        print("iteration:", iteration)
        for row in reversed(range(1, mdp.size[1] + 1)):
            for col in range(1, mdp.size[0] + 1):
                state = (col, row)
                if state in mdp.walls:
                    print("--------------", end="  ")
                elif state in mdp.terminal_states:
                    print("0", end="  ")  # Changed to always print 0 for terminal states
                else:
                    print(U[state], end="  ")
            print()
        print()

        for state in mdp.states:
            if mdp.is_terminal(state) or state in mdp.walls:
                continue
    
            max_q_value = float("-inf")
            for action in mdp.A(state):
                q_value = Q_value(mdp, state, action, U)
                if q_value > max_q_value:
                    max_q_value = q_value

            U_prime[state] = max_q_value
            diff = abs(U_prime[state] - U[state])

            if diff > delta:
                delta = diff

        if delta < epsilon * (1 - mdp.discount_rate) / mdp.discount_rate:
          print("Final Value After Convergence")
          for row in reversed(range(1, mdp.size[1] + 1)):
            for col in range(1, mdp.size[0] + 1):
              state = (col, row)
              if state in mdp.walls:
                  print("--------------", end="  ")
              elif state in mdp.terminal_states:
                  print("0", end="  ")  # Changed to always print 0 for terminal states
              else:
                  print(U[state], end="  ")
            print()
          print()
          break
        iteration += 1
    return U

def print_policy(policy, mdp):
    action_symbols = {(0, -1): 'S', (-1, 0): 'W', (0, 1): 'N', (1, 0): 'E'}
    print()
    for row in reversed(range(1, mdp.size[1] + 1)):
        for col in range(1, mdp.size[0] + 1):
            state = (col, row)
            if state in mdp.walls:
                print("-", end="  ")
            elif state in mdp.terminal_states:
                print("T", end="  ")
            else:
                print(action_symbols[policy[state]], end="  ")
        print()
    print()

def main():
    config = read_input('mdp_input.txt')
    size, walls, terminal_states, reward, transition_probabilities, discount_rate, epsilon = parse_input(config)
    mdp = GridworldMDP(size, walls, terminal_states, reward, transition_probabilities, discount_rate)
    print(f"({size[1]}, {size[0]}, [{', '.join([f'x={wall[0]} y={wall[1]}' for wall in walls])}], {({', '.join([f'x={state[0]} y={state[1]}: {value}' for state, value in terminal_states.items()])})}, {reward}, {transition_probabilities}, {discount_rate}, {epsilon})")
    print()
    U = value_iteration_v2(mdp, epsilon)
    policy = {state: None for state in mdp.states}

    for state in mdp.states:
        if mdp.is_terminal(state) or state in mdp.walls:
            continue

        max_q_value = float("-inf")
        best_action = None
        for action in mdp.A(state):
            q_value = Q_value(mdp, state, action, U)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
        policy[state] = best_action
    print()
    print("Final Policy")
    print_policy(policy, mdp)
    print("################ POLICY ITERATION ###########################")
    print_policy(policy, mdp)

if __name__ == "__main__":
    main()