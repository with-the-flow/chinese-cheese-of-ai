"""PyTorch 10层神经网络中国象棋AI"""
import numpy as np
import json
import time
import random
import os
from typing import Tuple, Dict, Any, List, Optional
from .engine import ChineseChess
import hashlib

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """10层全连接价值网络：输入棋盘特征，输出[-1, 1]的局面评分"""
    
    def __init__(self):
        super(ValueNetwork, self).__init__()
        # 网络结构: 105 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.layers = nn.Sequential(
            nn.Linear(105, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
            nn.Tanh(),  # 输出限制在[-1, 1]
        )
        
        # 初始化权重（接近0的小随机数，类似原NumPy版本）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ChessAI:
    """PyTorch版中国象棋AI（支持GPU加速与自动求导）"""
    
    def __init__(self, player: str, search_depth: int = 2, lr: float = 0.001,
                 device: Optional[torch.device] = None, shared_net: Optional[ValueNetwork] = None):
        self.player = player
        self.search_depth = search_depth
        self.lr = lr
        
        # 自动选择设备（CUDA > CPU）
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 网络实例：可共享（两个AI共用一套参数）
        if shared_net is not None:
            self.net = shared_net
        else:
            self.net = ValueNetwork().to(self.device)
        
        # 优化器由外部统一设置，防止共享网络时重复创建
        self.optimizer = None
        
        # 棋子权重（传统评估辅助）
        self.piece_weights = {1: 1000, 2: 20, 3: 20, 4: 40, 5: 90, 6: 45, 7: 10}
        
        # 缓存
        self._transposition_table: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._eval_count = 0
        self._start_time = 0.0
        
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """设置共享优化器"""
        self.optimizer = optimizer
    
    def _encode_board(self, board: np.ndarray, player: Optional[str] = None) -> np.ndarray:
        """将棋盘编码为105维特征向量（NumPy版本，供训练批量使用）"""
        if player is None:
            player = self.player
            
        encoded = np.zeros(105, dtype=np.float32)
        idx = 0
        
        # 1. 90个格子值
        encoded[idx:idx+90] = board.flatten().astype(np.float32)
        idx += 90
        
        # 2. 红方棋子数量特征（7种）
        for piece_type in range(1, 8):
            encoded[idx] = float(np.sum(board == piece_type))
            idx += 1
        
        # 3. 黑方棋子数量特征（7种）
        for piece_type in range(1, 8):
            encoded[idx] = float(np.sum(board == -piece_type))
            idx += 1
        
        # 4. 当前玩家特征
        encoded[idx] = 1.0 if player == 'red' else -1.0
        
        return encoded
    
    def _encode_board_tensor(self, board: np.ndarray, player: Optional[str] = None) -> torch.Tensor:
        """编码为PyTorch Tensor（单样本，推理用）"""
        encoded = self._encode_board(board, player)
        return torch.from_numpy(encoded).unsqueeze(0).to(self.device)  # shape (1, 105)
    
    def _compute_board_hash(self, board: np.ndarray) -> str:
        """计算棋盘哈希（用于缓存）"""
        return hashlib.md5(board.tobytes()).hexdigest()
    
    def evaluate_board(self, chess_game: ChineseChess, player: Optional[str] = None) -> float:
        """评估棋盘局面（PyTorch推理，无梯度，不构建计算图）"""
        if player is None:
            player = self.player
            
        board_hash = self._compute_board_hash(chess_game.board)
        cache_key = f"{board_hash}_{player}"
        
        if cache_key in self._transposition_table:
            self._cache_hits += 1
            return self._transposition_table[cache_key]
        
        self._cache_misses += 1
        self._eval_count += 1
        
        # PyTorch推理：关闭梯度，节省显存/内存
        self.net.eval()
        with torch.no_grad():
            x = self._encode_board_tensor(chess_game.board, player)
            neural_score = self.net(x).item()
        
        # 传统评估辅助
        piece_score = self._evaluate_pieces(chess_game, player)
        
        # 综合评分
        total_score = neural_score * 0.7 + piece_score * 0.3
        
        self._transposition_table[cache_key] = total_score
        return total_score
    
    def _evaluate_pieces(self, chess_game: ChineseChess, player: Optional[str] = None) -> float:
        """基于棋子价值的传统评估"""
        if player is None:
            player = self.player
            
        score = 0
        for i in range(10):
            for j in range(9):
                piece = chess_game.board[i, j]
                if piece != 0:
                    value = self.piece_weights.get(abs(piece), 0)
                    score += value if piece > 0 else -value
        
        if player == 'black':
            score = -score
        
        return score
    
    def get_best_move(self, chess_game: ChineseChess) -> Optional[Tuple]:
        """获取最佳走法（Minimax + Alpha-Beta）"""
        legal_moves = chess_game.get_legal_moves(self.player)
        if not legal_moves:
            return None
        
        random.shuffle(legal_moves)
        
        best_score = float('-inf')
        best_move = None
        self._start_time = time.time()
        
        for move in legal_moves:
            new_game = ChineseChess()
            new_game.board = chess_game.board.copy()
            new_game.current_player = chess_game.current_player
            new_game.game_over = chess_game.game_over
            new_game.winner = chess_game.winner
            new_game.make_move(move)
            
            score = self._minimax(new_game, self.search_depth - 1, float('-inf'), float('inf'), False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if len(self._transposition_table) > 100000:
            self._transposition_table.clear()
        
        return best_move
    
    def _minimax(self, chess_game: ChineseChess, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        """Minimax + Alpha-Beta剪枝"""
        board_hash = self._compute_board_hash(chess_game.board)
        cache_key = f"{board_hash}_{depth}_{maximizing_player}"
        
        if cache_key in self._transposition_table:
            return self._transposition_table[cache_key]
        
        if depth == 0 or chess_game.game_over:
            return self.evaluate_board(chess_game, self.player)
        
        if maximizing_player:
            max_eval = float('-inf')
            legal_moves = chess_game.get_legal_moves(self.player)
            
            for move in legal_moves:
                new_game = ChineseChess()
                new_game.board = chess_game.board.copy()
                new_game.current_player = chess_game.current_player
                new_game.game_over = chess_game.game_over
                new_game.winner = chess_game.winner
                new_game.make_move(move)
                
                eval_score = self._minimax(new_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break
            
            self._transposition_table[cache_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 'black' if self.player == 'red' else 'red'
            legal_moves = chess_game.get_legal_moves(opponent)
            
            for move in legal_moves:
                new_game = ChineseChess()
                new_game.board = chess_game.board.copy()
                new_game.current_player = chess_game.current_player
                new_game.game_over = chess_game.game_over
                new_game.winner = chess_game.winner
                new_game.make_move(move)
                
                eval_score = self._minimax(new_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break
            
            self._transposition_table[cache_key] = min_eval
            return min_eval
    
    def train_batch(self, boards: List[np.ndarray], targets: List[float], players: List[str]) -> float:
        """
        PyTorch批量训练：核心优势
        把一局所有步（通常30-80步）堆叠成一个batch，一次性forward+backward
        比NumPy逐样本训练快数十倍
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        # 编码所有棋盘状态
        encoded = np.array([self._encode_board(b, p) for b, p in zip(boards, players)], dtype=np.float32)
        
        x = torch.from_numpy(encoded).to(self.device)          # (batch_size, 105)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device).unsqueeze(1)  # (batch_size, 1)
        
        self.net.train()
        self.optimizer.zero_grad()
        pred = self.net(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        self.optimizer.step()
        
        # 权重已更新，清空缓存（旧评估失效）
        self._transposition_table.clear()
        
        return loss.item()
    
    def save_weights(self, filepath: str):
        """保存PyTorch权重（.pt格式）"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }, filepath)
    
    def load_weights(self, filepath: str):
        """加载PyTorch权重"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._transposition_table.clear()
        print(f"已加载权重: {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / max(total, 1),
            'eval_count': self._eval_count,
            'table_size': len(self._transposition_table),
            'device': str(self.device),
        }
