#!/usr/bin/env python3
"""后台训练脚本 - 持续运行版本"""
import sys
import os
import json
import time
import argparse
from datetime import datetime
from typing import Optional

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_core import ChineseChess, ChessAI
from chess_core.utils import get_cpu_usage, get_memory_usage, format_time

class SelfPlayTrainer:
    """自我对弈训练系统（持续运行版）"""
    
    def __init__(self, ai1: ChessAI, ai2: ChessAI, output_file: str = None):
        self.ai1 = ai1
        self.ai2 = ai2
        self.output_file = output_file
        self.training_data = []
        self.game_stats = {
            'total_games': 0,
            'red_wins': 0,
            'black_wins': 0,
            'total_moves': 0,
            'total_time': 0,
            'start_time': time.time(),
            'end_time': None,
            'stop_reason': 'unknown'
        }
        
        # 创建数据目录
        os.makedirs('data', exist_ok=True)
        
        # 如果没有指定输出文件，使用时间戳
        if not self.output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_file = f"data/training_{timestamp}.json"
        
    def run_self_play(self, duration: Optional[int] = None, max_games: Optional[int] = None):
        """
        运行自我对弈直到指定时间或局数
        
        Args:
            duration: 训练时长（秒），None表示无限
            max_games: 最大对局数，None表示无限
        """
        print("="*80)
        print("中国象棋AI自我对弈训练系统（持续运行版）")
        print("="*80)
        print(f"AI搜索深度: {self.ai1.search_depth}")
        print(f"输出文件: {self.output_file}")
        if duration:
            print(f"训练时长: {format_time(duration)}")
        if max_games:
            print(f"最大对局数: {max_games}")
        print("="*80)
        
        # 计算停止时间
        stop_time = time.time() + duration if duration else None
        
        try:
            game_id = 0
            while True:
                # 检查是否应停止
                if self._should_stop(stop_time, max_games, game_id):
                    self.training_stats['stop_reason'] = 'time_limit' if stop_time else 'game_limit'
                    break
                
                game_id += 1
                
                # 运行单局对弈
                game_start = time.time()
                game_data = self._play_single_game(game_id)
                game_duration = time.time() - game_start
                
                # 记录数据
                self.training_data.extend(game_data)
                self.game_stats['total_games'] += 1
                self.game_stats['total_moves'] += len(game_data)
                
                # 记录胜负
                result = game_data[-1]['result'] if game_data else 'draw'
                if result == 'red':
                    self.game_stats['red_wins'] += 1
                elif result == 'black':
                    self.game_stats['black_wins'] += 1
                
                # 每10局保存一次
                if game_id % 10 == 0:
                    self._save_data()
                    self._print_progress(game_id, game_duration, stop_time)
                
                # 休眠释放CPU
                time.sleep(0.005)  # 5ms
                
        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C，正在保存数据...")
            self.training_stats['stop_reason'] = 'user_interrupt'
        
        except Exception as e:
            print(f"\n\n发生错误: {e}")
            self.training_stats['stop_reason'] = 'error'
        
        finally:
            self.game_stats['end_time'] = time.time()
            self.game_stats['total_time'] = self.game_stats['end_time'] - self.game_stats['start_time']
            self._save_data()
            self._print_final_stats()
        
    def _should_stop(self, stop_time: Optional[float], max_games: Optional[int], current_games: int) -> bool:
        """检查是否应该停止训练"""
        # 检查时间限制
        if stop_time and time.time() >= stop_time:
            print("\n" + "="*80)
            print("训练时间已到，准备停止...")
            print("="*80)
            return True
        
        # 检查局数限制
        if max_games and current_games >= max_games:
            print("\n" + "="*80)
            print("已达到最大对局数，准备停止...")
            print("="*80)
            return True
        
        return False
    
    def _play_single_game(self, game_id: int) -> list:
        """单局对弈"""
        game = ChineseChess()
        game_data = []
        
        for move_count in range(1, 200):  # 最大200步防止死循环
            current_ai = self.ai1 if game.current_player == 'red' else self.ai2
            
            # 记录当前状态
            board_state = game.get_board_state().tolist()
            
            # AI选择走法
            move = current_ai.get_best_move(game)
            if not move:
                break
            
            # 执行走法
            game.make_move(move)
            
            # 记录数据（修复：转换为Python int）
            game_data.append({
                'game_id': game_id,
                'move_number': move_count,
                'player': game.current_player,
                'board_state': board_state,
                'move': move,
                'piece_type': int(abs(game.board[move[2], move[3]])),  # 修复
                'captured_piece': int(abs(game.board[move[2], move[3]])) if game.board[move[2], move[3]] != 0 else 0  # 修复
            })
            
            if game.game_over:
                break
        
        # 添加结果
        result = game.winner if game.winner else 'draw'
        for record in game_data:
            record['result'] = result
        
        return game_data
    
    def _save_data(self):
        """保存训练数据"""
        # 保存训练数据
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        with open("data/stats.json", 'w', encoding='utf-8') as f:
            json.dump(self.game_stats, f, ensure_ascii=False, indent=2)
            
        # 保存AI统计
        ai1_stats = self.ai1.get_stats()
        ai2_stats = self.ai2.get_stats()
        with open("data/ai_stats.json", 'w', encoding='utf-8') as f:
            json.dump({'ai1': ai1_stats, 'ai2': ai2_stats}, f, indent=2)
        
        # 清理AI缓存防止内存泄漏
        if len(self.ai1._transposition_table) > 50000:
            self.ai1._transposition_table.clear()
        if len(self.ai2._transposition_table) > 50000:
            self.ai2._transposition_table.clear()
    
    def _print_progress(self, current_games: int, last_game_time: float, stop_time: Optional[float]):
        """打印进度"""
        cpu_usage = get_cpu_usage()
        mem_usage = get_memory_usage()
        
        elapsed = time.time() - self.game_stats['start_time']
        remaining = max(stop_time - time.time(), 0) if stop_time else 0
        
        print(f"\n{'='*80}")
        print(f"已完成对局: {current_games}")
        print(f"最近一局: {last_game_time:.2f}秒")
        print(f"已运行: {format_time(elapsed)}")
        if stop_time:
            print(f"剩余时间: {format_time(remaining)}")
        print(f"红方胜率: {self.game_stats['red_wins']/current_games*100:.1f}%")
        print(f"黑方胜率: {self.game_stats['black_wins']/current_games*100:.1f}%")
        print(f"CPU使用率: {cpu_usage:.1f}%")
        print(f"内存使用: {mem_usage:.1f} MB")
        print(f"数据点数: {len(self.training_data)}")
        print(f"缓存命中率: {self.ai1.get_stats()['cache_hit_rate']:.2%}")
        print(f"{'='*80}\n")
    
    def _print_final_stats(self):
        """打印最终统计"""
        total_games = self.game_stats['total_games']
        total_time = self.game_stats['total_time']
        
        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)
        print(f"总对局数: {total_games}")
        print(f"总步数: {self.game_stats['total_moves']}")
        print(f"平均每局步数: {self.game_stats['total_moves']/total_games:.1f}")
        print(f"红方胜利: {self.game_stats['red_wins']} ({self.game_stats['red_wins']/total_games*100:.1f}%)")
        print(f"黑方胜利: {self.game_stats['black_wins']} ({self.game_stats['black_wins']/total_games*100:.1f}%)")
        print(f"停止原因: {self.game_stats['stop_reason']}")
        print(f"总用时: {format_time(total_time)}")
        print(f"输出文件: {self.output_file}")
        print("="*80)

def main():
    """主函数：启动训练"""
    parser = argparse.ArgumentParser(description="中国象棋AI后台训练程序")
    parser.add_argument(
        "--duration",
        type=int,
        help="训练时长（秒），默认无限",
        default=None
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="最大对局数，默认无限",
        default=None
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("中国象棋AI后台训练程序（持续运行版）")
    print("="*80)
    
    if args.duration:
        print(f"训练时长: {format_time(args.duration)}")
    if args.max_games:
        print(f"最大对局数: {args.max_games}")
    
    # 创建AI实例（深度1平衡速度和质量）
    ai1 = ChessAI('red', search_depth=1)
    ai2 = ChessAI('black', search_depth=1)
    
    # 创建训练器
    trainer = SelfPlayTrainer(ai1, ai2)
    
    # 开始训练
    trainer.run_self_play(duration=args.duration, max_games=args.max_games)
    
if __name__ == "__main__":
    main()