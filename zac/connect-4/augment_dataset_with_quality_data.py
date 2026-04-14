"""
AUGMENT EXISTING CONNECT 4 DATASET WITH HIGH-QUALITY DATA
==========================================================

This script:
1. Loads your existing cleaned dataset (X_train_cleaned.npy, y_train_cleaned.npy)
2. Generates NEW high-quality data using hybrid agent
3. Merges them together
4. Saves as X_train_final.npy and y_train_final.npy (overwrites old)

Strategy:
- Keep your existing 129K cleaned positions
- Add 50K-100K new high-quality positions
- Final dataset: 180K-230K positions with guaranteed tactical accuracy
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from pathlib import Path

# ============================================================================
# CONNECT 4 GAME ENGINE
# ============================================================================

class Connect4:
    """Connect 4 game engine"""
    
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=np.int8)
        self.current_player = 1
        
    def copy(self):
        """Create a deep copy of the game state"""
        new_game = Connect4()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game
    
    def get_legal_moves(self) -> List[int]:
        """Return list of legal column indices"""
        return [c for c in range(7) if self.board[0, c] == 0]
    
    def make_move(self, col: int) -> bool:
        """Make a move in the specified column"""
        if self.board[0, col] != 0:
            return False
        
        for row in range(5, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.current_player *= -1
                return True
        return False
    
    def check_win(self) -> Optional[int]:
        """Check if there's a winner"""
        # Horizontal
        for row in range(6):
            for col in range(4):
                if (self.board[row, col] != 0 and
                    self.board[row, col] == self.board[row, col+1] ==
                    self.board[row, col+2] == self.board[row, col+3]):
                    return self.board[row, col]
        
        # Vertical
        for row in range(3):
            for col in range(7):
                if (self.board[row, col] != 0 and
                    self.board[row, col] == self.board[row+1, col] ==
                    self.board[row+2, col] == self.board[row+3, col]):
                    return self.board[row, col]
        
        # Diagonal /
        for row in range(3, 6):
            for col in range(4):
                if (self.board[row, col] != 0 and
                    self.board[row, col] == self.board[row-1, col+1] ==
                    self.board[row-2, col+2] == self.board[row-3, col+3]):
                    return self.board[row, col]
        
        # Diagonal \
        for row in range(3):
            for col in range(4):
                if (self.board[row, col] != 0 and
                    self.board[row, col] == self.board[row+1, col+1] ==
                    self.board[row+2, col+2] == self.board[row+3, col+3]):
                    return self.board[row, col]
        
        return None
    
    def is_full(self) -> bool:
        """Check if board is full"""
        return np.all(self.board[0] != 0)
    
    def to_input_format(self) -> np.ndarray:
        """Convert to (6, 7, 2) format"""
        result = np.zeros((6, 7, 2), dtype=np.float32)
        result[:, :, 0] = (self.board == 1).astype(np.float32)
        result[:, :, 1] = (self.board == -1).astype(np.float32)
        return result


# ============================================================================
# TACTICAL HELPER FUNCTIONS
# ============================================================================

def find_immediate_win(game: Connect4, player: int) -> Optional[int]:
    """Find immediate winning move for player"""
    for col in game.get_legal_moves():
        test_game = game.copy()
        # Temporarily set player
        original_player = test_game.current_player
        test_game.current_player = player
        test_game.make_move(col)
        if test_game.check_win() == player:
            return col
        test_game.current_player = original_player
    return None


# ============================================================================
# SIMPLIFIED MINIMAX AGENT
# ============================================================================

class MinimaxAgent:
    """Minimax with alpha-beta pruning"""
    
    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth
    
    def evaluate(self, game: Connect4) -> float:
        """Simple evaluation"""
        winner = game.check_win()
        if winner == 1:
            return 1000000
        elif winner == -1:
            return -1000000
        return 0
    
    def minimax(self, game: Connect4, depth: int, alpha: float, 
                beta: float, maximizing: bool) -> Tuple[float, Optional[int]]:
        """Minimax search"""
        winner = game.check_win()
        if winner is not None:
            return (1000000 if winner == 1 else -1000000, None)
        
        if game.is_full() or depth == 0:
            return (self.evaluate(game), None)
        
        legal_moves = game.get_legal_moves()
        
        if maximizing:
            max_eval = float('-inf')
            best_move = legal_moves[0]
            
            for move in legal_moves:
                new_game = game.copy()
                new_game.make_move(move)
                eval_score, _ = self.minimax(new_game, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return (max_eval, best_move)
        else:
            min_eval = float('inf')
            best_move = legal_moves[0]
            
            for move in legal_moves:
                new_game = game.copy()
                new_game.make_move(move)
                eval_score, _ = self.minimax(new_game, depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return (min_eval, best_move)
    
    def get_move(self, game: Connect4) -> int:
        """Get best move"""
        _, best_move = self.minimax(
            game, self.max_depth, 
            float('-inf'), float('inf'),
            game.current_player == 1
        )
        return best_move


# ============================================================================
# SIMPLIFIED MCTS AGENT
# ============================================================================

class SimpleMCTSAgent:
    """MCTS with tactical override"""
    
    def __init__(self, simulations: int = 3000):
        self.simulations = simulations
    
    def get_move(self, game: Connect4) -> int:
        """Get move with tactical override"""
        
        # PRIORITY 1: Take winning move
        win_move = find_immediate_win(game, game.current_player)
        if win_move is not None:
            return win_move
        
        # PRIORITY 2: Block opponent
        block_move = find_immediate_win(game, -game.current_player)
        if block_move is not None:
            return block_move
        
        # PRIORITY 3: Use simple MCTS (random rollouts)
        legal_moves = game.get_legal_moves()
        
        # Simple UCB1 with random rollouts
        move_scores = {move: 0.0 for move in legal_moves}
        move_visits = {move: 0 for move in legal_moves}
        
        for _ in range(self.simulations):
            # Select move with UCB1
            best_ucb = float('-inf')
            best_move = legal_moves[0]
            total_visits = sum(move_visits.values()) + 1
            
            for move in legal_moves:
                if move_visits[move] == 0:
                    ucb = float('inf')
                else:
                    exploit = move_scores[move] / move_visits[move]
                    explore = 1.41 * np.sqrt(np.log(total_visits) / move_visits[move])
                    ucb = exploit + explore
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_move = move
            
            # Simulate from this move
            test_game = game.copy()
            test_game.make_move(best_move)
            
            # Random rollout
            while test_game.check_win() is None and not test_game.is_full():
                rand_move = random.choice(test_game.get_legal_moves())
                test_game.make_move(rand_move)
            
            # Backpropagate
            winner = test_game.check_win()
            if winner == game.current_player:
                reward = 1.0
            elif winner is None:
                reward = 0.5
            else:
                reward = 0.0
            
            move_scores[best_move] += reward
            move_visits[best_move] += 1
        
        # Return most visited move
        return max(move_visits.items(), key=lambda x: x[1])[0]


# ============================================================================
# HYBRID AGENT
# ============================================================================

class HybridAgent:
    """Hybrid: Minimax for early/late, MCTS for middle"""
    
    def __init__(self):
        self.minimax_agent = MinimaxAgent(max_depth=6)
        self.mcts_agent = SimpleMCTSAgent(simulations=3000)
    
    def get_move_count(self, game: Connect4) -> int:
        """Count pieces on board"""
        return int(np.sum(game.board != 0))
    
    def get_move(self, game: Connect4) -> int:
        """Choose agent based on game phase"""
        move_count = self.get_move_count(game)
        
        # Use minimax for opening and endgame
        if move_count < 10 or move_count > 27:
            return self.minimax_agent.get_move(game)
        else:
            return self.mcts_agent.get_move(game)


# ============================================================================
# DATASET AUGMENTATION
# ============================================================================

def generate_new_data(num_games: int = 3000, 
                      exploration_prob: float = 0.08,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate new high-quality data"""
    
    agent = HybridAgent()
    positions = []
    labels = []
    
    if verbose:
        print("="*70)
        print("GENERATING NEW HIGH-QUALITY DATA")
        print("="*70)
        print(f"Agent: Hybrid (Minimax + MCTS)")
        print(f"Games to generate: {num_games:,}")
        print(f"Exploration probability: {exploration_prob}")
        print()
    
    for game_num in range(num_games):
        if verbose and game_num % 10 == 0:
            print(f"Game {game_num}/{num_games} ({game_num/num_games*100:.1f}%) - " +
                  f"{len(positions):,} positions")
        
        game = Connect4()
        
        while game.check_win() is None and not game.is_full():
            # Get board state
            board_state = game.to_input_format()
            
            # Get move (with exploration)
            if random.random() < exploration_prob:
                move = random.choice(game.get_legal_moves())
            else:
                move = agent.get_move(game)
            
            # Store
            positions.append(board_state)
            labels.append(move)
            
            # Make move
            game.make_move(move)
    
    X_new = np.array(positions, dtype=np.float32)
    y_new = np.array(labels, dtype=np.int64)
    
    if verbose:
        print(f"\n✓ Generated {len(X_new):,} new positions from {num_games:,} games")
        print(f"  Average positions per game: {len(X_new)/num_games:.1f}")
    
    return X_new, y_new


def verify_tactical_quality(X: np.ndarray, y: np.ndarray, 
                            sample_size: int = 10000) -> dict:
    """Verify tactical correctness of dataset (uses find_immediate_win from this module)."""
    
    wins_found = 0
    wins_correct = 0
    blocks_found = 0
    blocks_correct = 0
    
    sample_size = min(sample_size, len(X))
    
    for i in range(sample_size):
        # Convert from (6,7,2) to game state for verification
        game = Connect4()
        game.board = X[i][:, :, 0] - X[i][:, :, 1]  # Reconstruct board
        
        # Determine current player
        player_pieces = X[i][:, :, 0].sum()
        ai_pieces = X[i][:, :, 1].sum()
        current_player = 1 if player_pieces == ai_pieces else -1
        
        # Check wins
        win_col = find_immediate_win(game, current_player)
        if win_col is not None:
            wins_found += 1
            if y[i] == win_col:
                wins_correct += 1
        
        # Check blocks
        block_col = find_immediate_win(game, -current_player)
        if block_col is not None:
            blocks_found += 1
            if y[i] == block_col:
                blocks_correct += 1
    
    return {
        'sample_size': sample_size,
        'wins_found': wins_found,
        'wins_correct': wins_correct,
        'blocks_found': blocks_found,
        'blocks_correct': blocks_correct
    }


def merge_datasets(X_old: np.ndarray, y_old: np.ndarray,
                   X_new: np.ndarray, y_new: np.ndarray,
                   shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Merge old and new datasets"""
    
    print("\n" + "="*70)
    print("MERGING DATASETS")
    print("="*70)
    print(f"Existing data: {len(X_old):,} positions")
    print(f"New data:      {len(X_new):,} positions")
    
    # Concatenate
    X_merged = np.concatenate([X_old, X_new], axis=0)
    y_merged = np.concatenate([y_old, y_new], axis=0)
    
    print(f"Merged data:   {len(X_merged):,} positions")
    
    # Shuffle
    if shuffle:
        print("Shuffling merged dataset...")
        indices = np.random.permutation(len(X_merged))
        X_merged = X_merged[indices]
        y_merged = y_merged[indices]
    
    return X_merged, y_merged


def print_dataset_stats(X: np.ndarray, y: np.ndarray, name: str = "Dataset"):
    """Print dataset statistics"""
    
    print(f"\n{name} Statistics:")
    print("-" * 70)
    
    # Column distribution
    print("\nColumn distribution:")
    for col in range(7):
        count = np.sum(y == col)
        pct = count / len(y) * 100
        print(f"  Column {col}: {count:,} ({pct:.1f}%)")
    
    # Balance check
    max_pct = max(np.sum(y == col) / len(y) for col in range(7))
    min_pct = min(np.sum(y == col) / len(y) for col in range(7))
    imbalance = (max_pct - min_pct) * 100
    print(f"\nColumn imbalance: {imbalance:.1f}% (lower is better)")
    
    print("-" * 70)


# ============================================================================
# DATASET CLEANING FUNCTIONS
# ============================================================================

def clean_existing_dataset(X: np.ndarray, y: np.ndarray, 
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Clean existing dataset by fixing tactical errors"""
    
    if verbose:
        print("="*70)
        print("CLEANING EXISTING DATASET")
        print("="*70)
        print(f"Original size: {len(X):,} positions")
        print()
    
    X_clean = []
    y_clean = []
    
    fixed_wins = 0
    fixed_blocks = 0
    removed_illegal = 0
    
    for i in range(len(X)):
        if verbose and i % 20000 == 0:
            print(f"Processing {i:,}/{len(X):,}...")
        
        # Convert board state to game for checking
        game = Connect4()
        game.board = X[i][:, :, 0] - X[i][:, :, 1]
        
        # Determine current player
        player_pieces = X[i][:, :, 0].sum()
        ai_pieces = X[i][:, :, 1].sum()
        current_player = 1 if player_pieces == ai_pieces else -1
        opponent = -current_player
        
        # Check for tactical corrections
        correct_label = None
        fix_type = None
        
        # PRIORITY 1: Check for immediate win
        win_col = find_immediate_win(game, current_player)
        if win_col is not None:
            correct_label = win_col
            fix_type = "win"
            if y[i] != win_col:
                fixed_wins += 1
        
        # PRIORITY 2: Check for immediate block (only if no win)
        if correct_label is None:
            block_col = find_immediate_win(game, opponent)
            if block_col is not None:
                correct_label = block_col
                fix_type = "block"
                if y[i] != block_col:
                    fixed_blocks += 1
        
        # PRIORITY 3: Check if current label is legal
        if correct_label is None:
            if X[i][0, y[i], 0] + X[i][0, y[i], 1] != 0:
                # Illegal move - skip
                removed_illegal += 1
                continue
            else:
                correct_label = y[i]
                fix_type = "original"
        
        # Add to cleaned dataset
        X_clean.append(X[i])
        y_clean.append(correct_label)
    
    if verbose:
        print()
        print("="*70)
        print("CLEANING RESULTS")
        print("="*70)
        print(f"Original examples:        {len(X):,}")
        print(f"Fixed winning moves:      {fixed_wins:,}")
        print(f"Fixed blocking moves:     {fixed_blocks:,}")
        print(f"Total fixed labels:       {fixed_wins + fixed_blocks:,} ({(fixed_wins + fixed_blocks)/len(X)*100:.1f}%)")
        print(f"Removed illegal moves:    {removed_illegal:,}")
        print(f"Final examples:           {len(X_clean):,}")
        print("="*70)
    
    return np.array(X_clean, dtype=np.float32), np.array(y_clean, dtype=np.int64)


# ============================================================================
# MAIN AUGMENTATION PIPELINE
# ============================================================================

def main():
    """Main augmentation pipeline with cleaning"""
    
    print("="*70)
    print("CONNECT 4 DATASET CLEANING & AUGMENTATION PIPELINE")
    print("="*70)
    print()
    
    # Step 1: Load existing data (uncleaned)
    print("Step 1: Loading existing dataset...")
    print("-" * 70)
    
    # Try to load existing data
    data_loaded = False
    
    # First try cleaned data (if already ran this script)
    try:
        X_old = np.load("X_train_cleaned.npy").astype("float32")
        y_old = np.load("y_train_cleaned.npy").astype("int64")
        print(f"✓ Loaded X_train_cleaned.npy: {len(X_old):,} positions")
        print("  (Already cleaned - skipping cleaning step)")
        data_loaded = True
        needs_cleaning = False
    except FileNotFoundError:
        pass
    
    # Then try original final data
    if not data_loaded:
        try:
            X_old = np.load("X_train_final.npy").astype("float32")
            y_old = np.load("y_train_final.npy").astype("int64")
            print(f"✓ Loaded X_train_final.npy: {len(X_old):,} positions")
            print("  (Needs cleaning)")
            data_loaded = True
            needs_cleaning = True
        except FileNotFoundError:
            pass
    
    if not data_loaded:
        print("✗ No dataset found!")
        print("\nPlease ensure one of these files exists:")
        print("  - X_train_final.npy and y_train_final.npy")
        print("  - X_train_cleaned.npy and y_train_cleaned.npy")
        return
    
    print_dataset_stats(X_old, y_old, "Original Dataset")
    
    # Step 2: Clean existing data (if needed)
    if needs_cleaning:
        print("\n" + "="*70)
        print("Step 2: Cleaning existing dataset...")
        print("-" * 70)
        
        X_old, y_old = clean_existing_dataset(X_old, y_old, verbose=True)
        
        # Save cleaned data for future use
        np.save("X_train_cleaned.npy", X_old)
        np.save("y_train_cleaned.npy", y_old)
        print(f"\n✓ Saved cleaned data for future use")
        
        # Verify cleaning quality
        print("\nVerifying cleaned data quality...")
        stats = verify_tactical_quality(X_old, y_old, sample_size=10000)
        
        print(f"Sample verification ({stats['sample_size']:,} positions):")
        if stats['wins_found'] > 0:
            print(f"  Wins: {stats['wins_correct']}/{stats['wins_found']} ({stats['wins_correct']/stats['wins_found']*100:.1f}%)")
        if stats['blocks_found'] > 0:
            print(f"  Blocks: {stats['blocks_correct']}/{stats['blocks_found']} ({stats['blocks_correct']/stats['blocks_found']*100:.1f}%)")
        
        if stats['wins_correct'] == stats['wins_found'] and \
           stats['blocks_correct'] == stats['blocks_found']:
            print("  ✅ PERFECT! Cleaned data has 100% tactical accuracy")
        
        print_dataset_stats(X_old, y_old, "Cleaned Dataset")
    else:
        print("\nStep 2: Cleaning existing dataset... SKIPPED (already clean)")
        print("-" * 70)
    
    # Step 3: Generate new high-quality data
    print("\n" + "="*70)
    print("Step 3: Generating new high-quality data...")
    print("-" * 70)
    
    # Generate 3000 games (~60K positions)
    X_new, y_new = generate_new_data(
        num_games=15000,
        exploration_prob=0.08,
        verbose=True
    )
    
    print_dataset_stats(X_new, y_new, "New Generated Data")
    
    # Step 4: Verify quality of new data
    print("\n" + "="*70)
    print("Step 4: Verifying tactical quality of new data...")
    print("-" * 70)
    
    stats = verify_tactical_quality(X_new, y_new, sample_size=10000)
    
    print(f"Sample size: {stats['sample_size']:,}")
    print(f"Winning positions: {stats['wins_found']:,}")
    print(f"  Correct: {stats['wins_correct']:,} ({stats['wins_correct']/stats['wins_found']*100:.1f}%)")
    print(f"Blocking positions: {stats['blocks_found']:,}")
    print(f"  Correct: {stats['blocks_correct']:,} ({stats['blocks_correct']/stats['blocks_found']*100:.1f}%)")
    
    if stats['wins_correct'] == stats['wins_found'] and \
       stats['blocks_correct'] == stats['blocks_found']:
        print("\n✅ PERFECT! New data has 100% tactical accuracy")
    else:
        print("\n⚠️  Warning: New data has tactical errors")
    
    # Step 5: Merge datasets
    X_final, y_final = merge_datasets(X_old, y_old, X_new, y_new, shuffle=True)
    
    print_dataset_stats(X_final, y_final, "Final Merged Dataset")
    
    # Step 6: Save final dataset
    print("\n" + "="*70)
    print("Step 6: Saving final dataset...")
    print("-" * 70)
    
    # Backup old files
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        shutil.copy("X_train_final.npy", f"X_train_final_backup_{timestamp}.npy")
        shutil.copy("y_train_final.npy", f"y_train_final_backup_{timestamp}.npy")
        print(f"✓ Backed up old files with timestamp: {timestamp}")
    except FileNotFoundError:
        print("  (No existing X_train_final.npy to backup)")
    
    # Save new final dataset
    np.save("X_train_final.npy", X_final)
    np.save("y_train_final.npy", y_final)
    
    print(f"\n✓ Saved X_train_final.npy ({X_final.nbytes / 1024**2:.1f} MB)")
    print(f"✓ Saved y_train_final.npy ({y_final.nbytes / 1024**2:.1f} MB)")
    
    # Summary
    print("\n" + "="*70)
    print("AUGMENTATION COMPLETE!")
    print("="*70)
    print(f"Original dataset:  {len(X_old):,} positions")
    print(f"New data added:    {len(X_new):,} positions")
    print(f"Final dataset:     {len(X_final):,} positions ({len(X_new)/len(X_old)*100:.1f}% increase)")
    print("\nYou can now retrain your models with the augmented dataset!")
    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run pipeline
    main()
