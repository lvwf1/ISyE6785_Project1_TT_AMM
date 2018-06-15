import  itertools

time_move = []
move = [-1, 0, 1]
for indices in itertools.product(range(len(tuple(move))), repeat=10):
    time_move.append([tuple(move)[i] for i in indices])


    [-1,-1,-1,-1]
    [-1,-1,0]