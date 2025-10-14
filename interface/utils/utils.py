
def make_titles(start, step, size):
    titles = []
    for i in range(size):
        for j in range(size):
            title = f"({start[0]+i*step[0]},{start[1]+j*step[1]})"
            titles.append(title)

    return titles
