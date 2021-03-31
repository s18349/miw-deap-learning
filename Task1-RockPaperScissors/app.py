
import numpy as np
import matplotlib.pyplot as plt


options = ['p', 's', 'r']

#start probabilities
start_prob = {
    'p': 0.33,
    's': 0.33,
    'r': 0.34
}
# learned probabilities
learned_prob = {
    'p': start_prob,
    's': start_prob,
    'r': start_prob
}

opponent_prob = {
    'p': {
        'p': 0.7,
        's': 0.2,
        'r': 0.1
    },
    's': {
        'p': 0.2,
        's': 0.1,
        'r': 0.7
    },
    'r': {
        'p': 0.2,
        's': 0.7,
        'r': 0.1
    }
}

learning_rate = 0.01
learned_prob_boundary = 0.1
precision = 4

opponent_last_move = np.random.choice(options, replace=False, p=list(start_prob.values()))
play_times = 3000
st = 1
score = 0
score_per_iteration = []
ratio_per_iteration = []

for i in range(play_times):
    opponent_new = np.random.choice(options, replace=False,
                              p=list(opponent_prob[opponent_last_move].values()))
    my_new = np.random.choice(options, replace=False,
                              p=list(learned_prob[opponent_last_move].values()))

    if opponent_new == 'p':
        if my_new == 's':  # we won
            ratio_per_iteration.append(1)
            score += 1

            if learned_prob[opponent_last_move][opponent_new] < (1 - learned_prob_boundary) and learned_prob[opponent_last_move]['r'] > learned_prob_boundary:
                learned_prob[opponent_last_move][opponent_new] += learning_rate
                learned_prob[opponent_last_move]['r'] -= learning_rate
        elif my_new == 'r':  # we lost
            ratio_per_iteration.append(-1)
            score -= 1

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['s'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['s'] += learning_rate
        elif my_new == 'p':  # tie
            ratio_per_iteration.append(0)

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['s'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['s'] += learning_rate

    if opponent_new == 's':
        if my_new == 'r':  # we won
            ratio_per_iteration.append(1)
            score += 1

            if learned_prob[opponent_last_move][opponent_new] < (1 - learned_prob_boundary) and learned_prob[opponent_last_move]['p'] > learned_prob_boundary:
                learned_prob[opponent_last_move][opponent_new] += learning_rate
                learned_prob[opponent_last_move]['p'] -= learning_rate
        elif my_new == 'p':  # we lost
            ratio_per_iteration.append(-1)
            score -= 1

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['r'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['r'] += learning_rate
        elif my_new == 's':  # tie
            ratio_per_iteration.append(0)

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['r'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['r'] += learning_rate

    if opponent_new == 'r':
        if my_new == 'p':  # we won
            ratio_per_iteration.append(1)
            score += 1

            if learned_prob[opponent_last_move][opponent_new] < (1 - learned_prob_boundary) and learned_prob[opponent_last_move]['s'] > learned_prob_boundary:
                learned_prob[opponent_last_move][opponent_new] += learning_rate
                learned_prob[opponent_last_move]['s'] -= learning_rate
        elif my_new == 's':  # we lost
            ratio_per_iteration.append(-1)
            score -= 1

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['p'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['p'] += learning_rate
        elif my_new == 'r':  # tie
            ratio_per_iteration.append(0)

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['p'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['p'] += learning_rate

    score_per_iteration.append(score)


plt.plot(ratio_per_iteration)
plt.plot(score_per_iteration)
plt.ylabel('score')
plt.xlabel('iteration')
plt.show()
