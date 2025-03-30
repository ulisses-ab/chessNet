import pandas as pd

chunk_size = 100000

df = pd.read_csv('games.csv', usecols=['moves', 'winner', 'victory_status'])

df = df.loc[df['victory_status'] != 'outoftime', ['winner', 'moves']]
df['winner'] = df['winner'].map({'white': 1, 'black': -1, 'draw': 0})

df.to_csv('clean_games.csv', index=False)