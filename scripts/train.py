#!/usr/bin/env python3
"""PyTorch后台训练脚本 - 单文件权重日志版"""
import sys
import os
import json
import time
import argparse
from typing import Optional
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_core import ChineseChess, ChessAI
from chess_core.utils import get_cpu_usage, get_memory_usage, format_time

import torch


class SelfPlayTrainer:
    """自我对弈训练系统（单文件权重日志版）"""
    
    def __init__(self, ai1: ChessAI, ai2: ChessAI, log_file: str = "train_data.log"):
        self.ai1 = ai1
        self.ai2 = ai2
        
        # 单文件日志路径（项目根目录）
        self.log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), log_file)
        print(f"权重日志文件: {self.log_file}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
        
        # 权重保存目录
        self.weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        print(f"权重保存到: {self.weights_dir}")
        
        # 共享优化器
        self.optimizer = torch.optim.Adam(ai1.net.parameters(), lr=ai1.lr)
        ai1.set_optimizer(self.optimizer)
        ai2.set_optimizer(self.optimizer)
        
        # 统计信息
        self.stats = {
            'total_games': 0,
            'red_wins': 0,
            'black_wins': 0,
            'total_moves': 0,
            'start_time': time.time(),
        }
        
    def run_self_play(self, max_games: int = 100000):
        """运行自我对弈训练"""
        print("=" * 80)
        print("中国象棋AI自我对弈训练系统（单文件日志版）")
        print(f"设备: {self.ai1.device}")
        print("=" * 80)
        print(f"AI搜索深度: {self.ai1.search_depth}")
        print(f"总训练局数: {max_games:,}")
        print(f"学习率: {self.ai1.lr}")
        print("=" * 80)
        
        try:
            for game_id in range(1, max_games + 1):
                game_start = time.time()
                game_states = self._play_single_game(game_id)
                game_duration = time.time() - game_start
                
                # 训练
                loss = 0.0
                if game_states:
                    loss = self._train_on_game(game_states)
                
                # 更新统计
                if game_states:
                    self.stats['total_games'] += 1
                    self.stats['total_moves'] += len(game_states)
                    
                    result = game_states[-1]['result']
                    if result == 'red':
                        self.stats['red_wins'] += 1
                    elif result == 'black':
                        self.stats['black_wins'] += 1
                
                # 🔥 每局结束：把当前权重扁平化为一行，追加写入日志
                self._log_weights(game_id, loss if game_states else None, 
                                  game_states[-1]['result'] if game_states else 'draw',
                                  len(game_states) if game_states else 0)
                
                # 每10000局保存一次 .pt 权重文件（用于恢复训练）
                if game_id % 10000 == 0:
                    self._save_weights(game_id)
                    self._print_progress(game_id, game_duration, loss if game_states else None)
                
                time.sleep(0.005)
                
        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C，正在保存...")
            self._save_weights(self.stats['total_games'])
        
        except Exception as e:
            print(f"\n\n发生错误: {e}")
            import traceback
            traceback.print_exc()
            self._save_weights(self.stats['total_games'])
        
        finally:
            self.stats['end_time'] = time.time()
            self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
            self._print_final_stats()
    
    def _play_single_game(self, game_id: int) -> list:
        """单局对弈"""
        game = ChineseChess()
        game_states = []
        
        for move_count in range(1, 200):
            current_ai = self.ai1 if game.current_player == 'red' else self.ai2
            
            board_state = game.get_board_state().tolist()
            current_player = game.current_player
            
            move = current_ai.get_best_move(game)
            if not move:
                break
            
            x1, y1, x2, y2 = move
            piece_type = int(abs(game.board[x1, y1]))
            captured_piece = int(abs(game.board[x2, y2])) if game.board[x2, y2] != 0 else 0
            
            game.make_move(move)
            
            game_states.append({
                'game_id': game_id,
                'move_number': move_count,
                'player': current_player,
                'board_state': board_state,
                'move': move,
                'piece_type': piece_type,
                'captured_piece': captured_piece,
            })
            
            if game.game_over:
                break
        
        result = game.winner if (game.game_over and game.winner) else 'draw'
        for state in game_states:
            state['result'] = result
        
        return game_states
    
    def _train_on_game(self, game_states: list) -> float:
        """PyTorch批量训练"""
        if not game_states:
            return 0.0
        
        result = game_states[-1]['result']
        
        boards = []
        targets = []
        players = []
        
        for state in game_states:
            player = state['player']
            board = np.array(state['board_state'], dtype=np.int8)
            
            boards.append(board)
            players.append(player)
            
            if result == 'draw':
                target = 0.0
            elif result == 'red':
                target = 1.0 if player == 'red' else -1.0
            else:
                target = -1.0 if player == 'red' else 1.0
            targets.append(target)
        
        return self.ai1.train_batch(boards, targets, players)
    
    def _log_weights(self, game_id: int, loss: Optional[float], result: str, move_count: int):
        """
        🔥 核心：每局结束后，把当前网络权重扁平化为一行JSON，追加写入日志
        """
        # 提取所有权重参数，扁平化为列表
        weights_flat = []
        for name, param in self.ai1.net.named_parameters():
            weights_flat.extend(param.detach().cpu().numpy().flatten().tolist())
        
        # 构建日志行
        log_entry = {
            'game_id': game_id,
            'timestamp': time.time(),
            'loss': round(loss, 6) if loss is not None else None,
            'result': result,
            'move_count': move_count,
            'red_wins': self.stats['red_wins'],
            'black_wins': self.stats['black_wins'],
            'total_games': self.stats['total_games'],
            'weights': weights_flat,  # 扁平化的所有权重
        }
        
        # 追加写入（每行一条JSON）
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def _save_weights(self, game_id: int):
        """保存PyTorch权重（用于断点恢复）"""
        filepath = os.path.join(self.weights_dir, f'weights_{game_id}.pt')
        self.ai1.save_weights(filepath)
        print(f"✓ 已保存权重: {filepath}")
    
    def _print_progress(self, current_games: int, last_game_time: float, loss: Optional[float]):
        """打印进度"""
        cpu_usage = get_cpu_usage()
        mem_usage = get_memory_usage()
        elapsed = time.time() - self.stats['start_time']
        
        print(f"\n{'=' * 80}")
        print(f"进度: {current_games:,} 局")
        print(f"最近一局: {last_game_time:.2f}秒")
        if loss is not None:
            print(f"最近Loss: {loss:.6f}")
        print(f"已运行: {format_time(elapsed)}")
        total = self.stats['total_games']
        if total > 0:
            print(f"红方胜率: {self.stats['red_wins']/total*100:.1f}%")
            print(f"黑方胜率: {self.stats['black_wins']/total*100:.1f}%")
            print(f"平均步数: {self.stats['total_moves']/total:.1f}")
        print(f"CPU使用率: {cpu_usage:.1f}%")
        print(f"内存使用: {mem_usage:.1f} MB")
        print(f"设备: {self.ai1.device}")
        print(f"日志大小: {os.path.getsize(self.log_file) / 1024 / 1024:.2f} MB")
        print(f"{'=' * 80}\n")
    
    def _print_final_stats(self):
        """打印最终统计"""
        total_games = self.stats['total_games']
        if total_games == 0:
            print("没有完成对局")
            return
        
        total_time = self.stats['total_time']
        log_size_mb = os.path.getsize(self.log_file) / 1024 / 1024 if os.path.exists(self.log_file) else 0
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        print(f"总对局数: {total_games:,}")
        print(f"总用时: {format_time(total_time)}")
        print(f"平均每局时间: {total_time/total_games:.2f}秒")
        print(f"日志文件: {self.log_file}")
        print(f"日志大小: {log_size_mb:.2f} MB")
        print(f"总行数: {total_games} 行（每局一行）")
        print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中国象棋AI PyTorch训练程序（单文件日志版）")
    parser.add_argument("--max-games", type=int, default=100000, help="每轮训练局数")
    parser.add_argument("--rounds", type=int, default=1, help="训练轮数")
    parser.add_argument("--load-weights", type=str, default=None, help="加载预训练权重(.pt)")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--device", type=str, default=None, help="cpu 或 cuda")
    parser.add_argument("--log-file", type=str, default="train_data.log", help="权重日志文件名")
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    print("=" * 80)
    print("中国象棋AI PyTorch训练程序（单文件日志版）")
    print("=" * 80)
    print(f"每轮局数: {args.max_games:,}")
    print(f"总轮数: {args.rounds}")
    print(f"学习率: {args.lr}")
    print(f"日志文件: {args.log_file}")
    if device:
        print(f"指定设备: {device}")
    print("=" * 80)
    
    ai1 = ChessAI('red', search_depth=1, lr=args.lr, device=device)
    ai2 = ChessAI('black', search_depth=1, lr=args.lr, device=device, shared_net=ai1.net)
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'=' * 80}")
        print(f"第 {round_num}/{args.rounds} 轮训练开始")
        print(f"{'=' * 80}")
        
        weights_to_load = None
        if round_num > 1:
            prev_file = os.path.join("weights", "weights_final.pt")
            if os.path.exists(prev_file):
                weights_to_load = prev_file
        elif args.load_weights and os.path.exists(args.load_weights):
            weights_to_load = args.load_weights
        
        if weights_to_load:
            print(f"加载预训练权重: {weights_to_load}")
            ai1.load_weights(weights_to_load)
        
        trainer = SelfPlayTrainer(ai1, ai2, log_file=args.log_file)
        trainer.run_self_play(max_games=args.max_games)
        
        final_path = os.path.join("weights", "weights_final.pt")
        ai1.save_weights(final_path)
        print(f"\n第 {round_num} 轮完成，权重已保存: {final_path}")
    
    print(f"\n{'=' * 80}")
    print(f"全部 {args.rounds} 轮训练完成！")
    print(f"最终权重: weights/weights_final.pt")
    print(f"权重日志: {args.log_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
