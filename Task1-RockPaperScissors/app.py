
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
play_times = 2000
score = 0
score_per_round = []
ratio_per_round = []

for i in range(play_times):
    opponent_new = np.random.choice(options, replace=False,
                              p=list(opponent_prob[opponent_last_move].values()))
    my_new = np.random.choice(options, replace=False,
                              p=list(learned_prob[opponent_last_move].values()))

    if opponent_new == 'p':
        if my_new == 's':  # when we won
            ratio_per_round.append(1)
            score += 1

            if learned_prob[opponent_last_move][opponent_new] < (1 - learned_prob_boundary) and learned_prob[opponent_last_move]['r'] > learned_prob_boundary:
                learned_prob[opponent_last_move][opponent_new] += learning_rate
                learned_prob[opponent_last_move]['r'] -= learning_rate
        elif my_new == 'r':  # when we lost
            ratio_per_round.append(-1)
            score -= 1

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['s'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['s'] += learning_rate
        elif my_new == 'p':  # when we tie
            ratio_per_round.append(0)

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['s'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['s'] += learning_rate

    if opponent_new == 's':
        if my_new == 'r':  # we when we won
            ratio_per_round.append(1)
            score += 1

            if learned_prob[opponent_last_move][opponent_new] < (1 - learned_prob_boundary) and learned_prob[opponent_last_move]['p'] > learned_prob_boundary:
                learned_prob[opponent_last_move][opponent_new] += learning_rate
                learned_prob[opponent_last_move]['p'] -= learning_rate
        elif my_new == 'p':  # when we  lost
            ratio_per_round.append(-1)
            score -= 1

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['r'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['r'] += learning_rate
        elif my_new == 's':  #When tie
            ratio_per_round.append(0)

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['r'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['r'] += learning_rate

    if opponent_new == 'r':
        if my_new == 'p':  #When we won
            ratio_per_round.append(1)
            score += 1

            if learned_prob[opponent_last_move][opponent_new] < (1 - learned_prob_boundary) and learned_prob[opponent_last_move]['s'] > learned_prob_boundary:
                learned_prob[opponent_last_move][opponent_new] += learning_rate
                learned_prob[opponent_last_move]['s'] -= learning_rate
        elif my_new == 's':  #When we lost
            ratio_per_round.append(-1)
            score -= 1

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['p'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['p'] += learning_rate
        elif my_new == 'r':  # when we tie
            ratio_per_round.append(0)

            if learned_prob[opponent_last_move][opponent_new] > learned_prob_boundary and learned_prob[opponent_last_move]['p'] < (1 - learned_prob_boundary):
                learned_prob[opponent_last_move][opponent_new] -= learning_rate
                learned_prob[opponent_last_move]['p'] += learning_rate

    score_per_round.append(score)


plt.plot(ratio_per_round)
plt.plot(score_per_round)
plt.ylabel('score')
plt.xlabel('round')
plt.show()
