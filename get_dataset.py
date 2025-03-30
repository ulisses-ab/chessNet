import pandas as pd
import chess
import torch
import random

def main():
    df = pd.read_csv('clean_games.csv')[:5000]
    states, results = tensors_from_df(df)
    dataset = torch.utils.data.TensorDataset(states, results)
    torch.save(dataset, 'dataset.pt')

def tensors_from_df(df):
    recordings = {}
    store_df_on_dict(recordings, df)

    states = torch.zeros(len(recordings), 15, 8, 8)
    results = torch.zeros(len(recordings), 1)
    for i, item in enumerate(recordings.items()):
        if(i % 100 == 0):
            print(f'Processing item: {i}')

        state, result = tensors_from_item(item)
        states[i] = state
        results[i] = result

    return states, results

def store_df_on_dict(dict, df):
    for i, row in df.iterrows():
        if(i % 100 == 0):
            print(f'Storing row: {i}')

        store_row_on_dict(dict, row)

def store_row_on_dict(dict, row):
    moves = row['moves'].split(sep=' ')
    winner = row['winner']
    board = chess.Board()

    for i, move in enumerate(moves):
        board.push_san(move)

        fen = board.fen()
        if fen in dict:
            dict[fen][winner + 1] += 1
        else:
            results = [0, 0, 0]
            results[winner + 1] = 1
            dict[fen] = results

def tensors_from_item(item):
    key, value = item

    board = chess.Board(key)

    tensor = board_to_tensor(board)
    result = value[2] / sum(value) - value[0] / sum(value)

    return tensor, torch.tensor(result).unsqueeze(0)
    
def board_to_tensor(board):
    board_tensor = torch.zeros(15, 8, 8)
    
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color = piece.color
        board_tensor[piece_type - 1 + (color * 6), chess.square_rank(square), chess.square_file(square)] = 1

    board_tensor[12, :, :] = board.turn
    board_tensor[13, :, :] = board.is_check()
    board_tensor[14, :4, :4] = board.has_kingside_castling_rights(chess.WHITE)
    board_tensor[14, 4:, :4] = board.has_queenside_castling_rights(chess.WHITE)
    board_tensor[14, :4, 4:] = board.has_kingside_castling_rights(chess.BLACK)
    board_tensor[14, 4:, 4:] = board.has_queenside_castling_rights(chess.BLACK)

    return board_tensor

if __name__ == "__main__":
    main()