import sys

from Player import *
from QTrainer import *
from Learning import *
from EvaluationFunctions import *

PLAYERS_TYPES = {
    0: 'HUMAN_PLAYER1',
    1: 'RANDOM_PLAYER1',
    2: 'MINIMAX_PLAYER1',
    3: 'ABPRUNE_PLAYER1',
    4: 'QPLAYER1',
    5: 'MIX'
}

VALUE_FUNCTION_DICT = {
    0: EvaluationFunctions.null_heuristic,
    1: EvaluationFunctions.points_diff_heuristic,
    2: EvaluationFunctions.max_my_points,
    3: EvaluationFunctions.min_opp_points,
    4: EvaluationFunctions.max_repeat_turns,
    5: EvaluationFunctions.max_my_playing_marbles,
    6: EvaluationFunctions.mixed_heuristic
}

PRINT_MESSAGE = "in testing {0} won {1} games and {2} won {3} games\n"
PRINT_MESSAGE_WITH_TIES = "in testing {0} won {1} games, {2} won {3} games " \
                          "and there were {4} ties\n"

def q_learning_training(player, opponent):
    if hasattr(player, 'epsilon'):
        QLearner_one = QLearningAgent(player, EvaluationFunctions.oneExtractor(player),
                                      player.depth, player.epsilon, player.alpha,
                                      player.gamma)
    else:
        QLearner_one = QLearningAgent(player, EvaluationFunctions.oneExtractor(player),
                                      player.depth)
    player.agent = QLearner_one
    trainer = QTrainer(player, opponent)
    trainer.train()
    return player

def q_learning_suitability(player, opponent):
    if player.type == Player.CUSTOM:
        player = q_learning_training(player, opponent)
    return player

def run_game(first_player, second_player, rounds):
    first_player = q_learning_suitability(first_player, second_player)
    second_player = q_learning_suitability(second_player, first_player)
    if Player.HUMAN in {first_player.type, second_player.type}:
        start_game_function = MancalaGUI.startGame
    else:
        start_game_function = MancalaGUI.startGameNoGUI

    plays = util.Counter()
    for i in range(rounds):
        plays[start_game_function(first_player, second_player, i)] += 1
    results_print(first_player, str(plays[1]), second_player, str(plays[2]),
                  str(plays[0]))

def results_print(first_player, first_points, second_player, second_points,
                  ties_amount):
    if not ties_amount:
        print(PRINT_MESSAGE.format(first_player, first_points, second_player,
                                   second_points))
    else:
        print(PRINT_MESSAGE_WITH_TIES.format(first_player, first_points,
                                             second_player, second_points,
                                             ties_amount))


def main(first_player, first_value_func, first_depth, second_player,
         second_value_func, second_depth, rounds, epsilon, alpha, gamma):
    first_value_func = VALUE_FUNCTION_DICT[int(first_value_func)]
    second_value_func = VALUE_FUNCTION_DICT[int(second_value_func)]
    first_player_obj = Player(1, int(first_player), int(second_player),
                              int(first_depth), first_value_func)
    second_player_obj = Player(2, int(second_player), int(first_player),
                               int(second_depth), second_value_func,)
    if epsilon is not None:
        epsilon = float(epsilon)
        alpha = float(alpha)
        gamma = float(gamma)
        if first_player_obj.type == Player.CUSTOM:
            first_player_obj.set_learning_values(epsilon, alpha, gamma)
        if second_player_obj.type == Player.CUSTOM:
            second_player_obj.set_learning_values(epsilon, alpha, gamma)

    rounds = int(rounds)
    run_game(first_player_obj, second_player_obj, rounds)


# qlearner diff vals vs alpha beta
def research():
    for a in [0.8, 0.6, 0.4]:
        for f1 in range(1, len(VALUE_FUNCTION_DICT)):
            for f2 in range(1, len(VALUE_FUNCTION_DICT)):
                player_obj_1 = Player(1, 4, 3, 100,
                                      VALUE_FUNCTION_DICT[f1])
                player_obj_1.set_learning_values(0.05, a, 0.8)
                player_obj_2 = Player(2, 3, 4, 2,
                                      VALUE_FUNCTION_DICT[f2])
                summary_print(
                    [4, f1, 100, 3, f2, 2, 100, None, a, None])
                run_game(player_obj_1, player_obj_2, 100)

# function for finding optimal discount factor and reword
# def find_optimal_weights(rounds):
#     q_learners = []
#     result = 0
#     for i in range(0, 11):
#         summary_print([4, 1, 200, 4, 1, 200, 100, 0.05, 1, i/10, 0.05, 1, (i+1)/10])
#         main(4, 1, 200, 4, 1, 200, 100, 0.05, 1, i/10, 0.05, 1, (i+1)/10)

# alpha beta vs alpha beta
# def research():
#     for depth in range(5, 7):
#         visited = []
#         for f1 in range(1, len(VALUE_FUNCTION_DICT)):
#             visited.append(f1)
#             for f2 in range(1, len(VALUE_FUNCTION_DICT)):
#                 if f1 != f2 and f2 not in visited:
#                     player_obj_1 = Player(1, 3, 3, depth,
#                                           VALUE_FUNCTION_DICT[f1])
#                     player_obj_2 = Player(2, 3, 3, depth,
#                                           VALUE_FUNCTION_DICT[f2])
#                     summary_print(
#                         [3, f1, depth, 3, f2, depth, 10, None, None, None])
#                     run_game(player_obj_1, player_obj_2, 10)

# alpha beta vs q-learner
# def research():
#     for depth in [100, 1000, 5000]:
#         for f1 in range(1, len(VALUE_FUNCTION_DICT)):
#             for f2 in range(1, len(VALUE_FUNCTION_DICT)):
#                 player_obj_1 = Player(1, 3, 4, 2,
#                                       VALUE_FUNCTION_DICT[f1])
#                 player_obj_2 = Player(2, 4, 3, depth,
#                                       VALUE_FUNCTION_DICT[f2])
#                 summary_print(
#                     [3, f1, 2, 4, f2, depth, 100, None, None, None])
#                 run_game(player_obj_1, player_obj_2, 100)

# # alpha beta vs mixed
# def research():
#     for depth in range(1, 5):
#         for f1 in range(1, len(VALUE_FUNCTION_DICT)):
#             player_obj_1 = Player(1, 3, 1, depth,
#                                   VALUE_FUNCTION_DICT[f1])
#             player_obj_2 = Player(2, 1, 3)
#             summary_print(
#                 [3, f1, depth, 1, 0, 0, 10, None, None, None])
#             run_game(player_obj_1, player_obj_2, 10)

# def research():
#     summary_print([3, 1, 2, 3, 1, 1, 100, None, None, None, None, None, None])
#     main(4, 1, 2, 4, 1, 1, 100, 0.05, 1, 0.1, 0.05, 1, 0.2)
#     summary_print([3, 1, 2, 3, 1, 1, 50, None, None, None, None, None, None])
#     main(3, 1, 2, 3, 1, 1, 50, None, None, None, None, None, None)

def summary_print(arguments):
    args_titles = [
        'first type num',
        'first value func',
        'first tree search depth\learning rounds amount',
        'second type num',
        'second value func',
        'second tree search depth\learning rounds amount',
        'game rounds',
        'epsilon',
        'alpha',
        'gamma',
    ]
    for i, arg in enumerate(arguments):
        print('{0}: {1}'.format(args_titles[i], arg))

if __name__ == '__main__':
    if sys.argv[1] == 'research':
        research()
    elif len(sys.argv) != 8 and len(sys.argv) != 11:
        print("Arguments format is: first player type num, "
              "first player value func, tree search depth\\learning rounds amount, "
              + "second player type...\nit's possible to add epsilon, alpha & "
                "gamma at the end (10 args in total instead of 7)")
    elif sys.argv[3] == '0' or sys.argv[6] == '0':
        print("Oh you didn't!!!! tree search depth\\learning rounds amount of "
              "size 0 are big NO NO")
    else:
        args = sys.argv[1:]
        summary_print(args)
        if len(sys.argv) == 8:
            args += [None, None, None]
        main(*args)

