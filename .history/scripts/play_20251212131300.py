#!/usr/bin/env python3
# play.py - 人对战程序（带图形界面）
import pygame
import sys, os
# 确保脚本目录与父目录被加入模块搜索路径（按需启用）
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_this_dir, ".."))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# 现在安全导入
from chess_core import ChineseChess, ChessAI



class ChessGUI:
    """中国象棋图形界面"""
    
    def __init__(self, ai_depth: int = 3):
        pygame.init()
        self.width, self.height = 720, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("中国象棋 - 人机对战")
        
        self.chess_game = ChineseChess()
        self.ai = ChessAI('black', search_depth=ai_depth)  # AI执黑方
        
        self.cell_size = 70
        self.board_margin_x = (self.width - 8 * self.cell_size) // 2
        self.board_margin_y = (self.height - 9 * self.cell_size) // 2
        
        self.selected_piece = None
        self.valid_moves = []
        
        # 颜色定义
        self.bg_color = (240, 217, 181)
        self.board_color = (210, 180, 140)
        self.line_color = (0, 0, 0)
        self.red_color = (255, 0, 0)
        self.black_color = (0, 0, 0)
        self.hint_color = (0, 255, 0)
        
        self.font = pygame.font.SysFont('simhei', 24)
        self.small_font = pygame.font.SysFont('simhei', 18)
        
        self.game_mode = 'human_vs_ai'  # 人对AI模式
    
    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(self.bg_color)
        
        # 绘制棋盘背景
        board_rect = pygame.Rect(
            self.board_margin_x - 10,
            self.board_margin_y - 10,
            8 * self.cell_size + 20,
            9 * self.cell_size + 20
        )
        pygame.draw.rect(self.screen, self.board_color, board_rect)
        
        # 绘制横线
        for i in range(10):
            y = self.board_margin_y + i * self.cell_size
            pygame.draw.line(
                self.screen, self.line_color,
                (self.board_margin_x, y),
                (self.board_margin_x + 8 * self.cell_size, y),
                2
            )
        
        # 绘制竖线
        for i in range(9):
            x = self.board_margin_x + i * self.cell_size
            if i == 0 or i == 8:
                pygame.draw.line(
                    self.screen, self.line_color,
                    (x, self.board_margin_y),
                    (x, self.board_margin_y + 9 * self.cell_size),
                    2
                )
            else:
                pygame.draw.line(
                    self.screen, self.line_color,
                    (x, self.board_margin_y),
                    (x, self.board_margin_y + 4 * self.cell_size),
                    2
                )
                pygame.draw.line(
                    self.screen, self.line_color,
                    (x, self.board_margin_y + 5 * self.cell_size),
                    (x, self.board_margin_y + 9 * self.cell_size),
                    2
                )
        
        # 绘制九宫斜线
        # 红方九宫
        pygame.draw.line(
            self.screen, self.line_color,
            (self.board_margin_x + 3 * self.cell_size, self.board_margin_y + 7 * self.cell_size),
            (self.board_margin_x + 5 * self.cell_size, self.board_margin_y + 9 * self.cell_size),
            2
        )
        pygame.draw.line(
            self.screen, self.line_color,
            (self.board_margin_x + 5 * self.cell_size, self.board_margin_y + 7 * self.cell_size),
            (self.board_margin_x + 3 * self.cell_size, self.board_margin_y + 9 * self.cell_size),
            2
        )
        
        # 黑方九宫
        pygame.draw.line(
            self.screen, self.line_color,
            (self.board_margin_x + 3 * self.cell_size, self.board_margin_y),
            (self.board_margin_x + 5 * self.cell_size, self.board_margin_y + 2 * self.cell_size),
            2
        )
        pygame.draw.line(
            self.screen, self.line_color,
            (self.board_margin_x + 5 * self.cell_size, self.board_margin_y),
            (self.board_margin_x + 3 * self.cell_size, self.board_margin_y + 2 * self.cell_size),
            2
        )
    
    def draw_pieces(self):
        """绘制棋子"""
        piece_names = {
            1: '帅', -1: '将',
            2: '仕', -2: '士', 
            3: '相', -3: '象',
            4: '马', -4: '马',
            5: '车', -5: '车',
            6: '炮', -6: '炮',
            7: '兵', -7: '卒'
        }
        
        for i in range(10):
            for j in range(9):
                piece = self.chess_game.board[i, j]
                if piece != 0:
                    x = self.board_margin_x + j * self.cell_size
                    y = self.board_margin_y + i * self.cell_size
                    
                    # 绘制棋子圆形背景
                    color = self.red_color if piece > 0 else self.black_color
                    pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 30)
                    pygame.draw.circle(self.screen, color, (x, y), 30, 2)
                    
                    # 绘制棋子文字
                    text = self.font.render(piece_names[piece], True, color)
                    text_rect = text.get_rect(center=(x, y))
                    self.screen.blit(text, text_rect)
    
    def draw_valid_moves(self):
        """绘制合法走法提示"""
        for move in self.valid_moves:
            x1, y1, x2, y2 = move
            if (x1, y1) == self.selected_piece:
                target_x = self.board_margin_x + y2 * self.cell_size
                target_y = self.board_margin_y + x2 * self.cell_size
                pygame.draw.circle(self.screen, self.hint_color, (target_x, target_y), 10)
    
    def draw_info(self):
        """绘制游戏信息"""
        # 当前玩家信息
        player_text = f"当前回合: {'红方（你）' if self.chess_game.current_player == 'red' else '黑方（AI）'}"
        text_surface = self.small_font.render(player_text, True, self.line_color)
        self.screen.blit(text_surface, (20, 20))
        
        # 游戏状态信息
        if self.chess_game.game_over:
            status_text = f"游戏结束! 胜利方: {self.chess_game.winner}"
            color = self.red_color if self.chess_game.winner == 'red' else self.black_color
            status_surface = self.small_font.render(status_text, True, color)
            self.screen.blit(status_surface, (20, 50))
        
        # 操作提示
        hint_text = "点击红方棋子选择，再点击绿色提示落子"
        hint_surface = self.small_font.render(hint_text, True, self.line_color)
        self.screen.blit(hint_surface, (20, 80))
        
        # AI难度
        ai_text = f"AI搜索深度: {self.ai.search_depth}"
        ai_surface = self.small_font.render(ai_text, True, self.line_color)
        self.screen.blit(ai_surface, (20, 110))
    
    def get_board_position(self, mouse_pos: Tuple) -> Tuple:
        """将鼠标位置转换为棋盘坐标"""
        x, y = mouse_pos
        board_x = (x - self.board_margin_x) // self.cell_size
        board_y = (y - self.board_margin_y) // self.cell_size
        
        if 0 <= board_x < 9 and 0 <= board_y < 10:
            return (board_y, board_x)
        return None
    
    def run(self):
        """运行游戏主循环"""
        clock = pygame.time.Clock()
        running = True
        
        print("游戏开始！你是红方，先手。")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.chess_game.game_over:
                        pos = self.get_board_position(event.pos)
                        if pos:
                            r, c = pos
                            
                            # 红方回合（玩家）
                            if self.chess_game.current_player == 'red':
                                if self.selected_piece is None:
                                    # 选择棋子
                                    if self.chess_game.board[r, c] > 0:
                                        self.selected_piece = (r, c)
                                        self.valid_moves = self.chess_game.get_legal_moves('red')
                                        self.valid_moves = [m for m in self.valid_moves if m[0] == r and m[1] == c]
                                else:
                                    # 落子
                                    for move in self.valid_moves:
                                        if move[2] == r and move[3] == c:
                                            self.chess_game.make_move(move)
                                            print(f"你的走法: {move}")
                                            self.selected_piece = None
                                            self.valid_moves = []
                                            break
                                    else:
                                        # 点击无效位置，清空选择
                                        self.selected_piece = None
                                        self.valid_moves = []
            
            # AI回合（黑方）
            if not self.chess_game.game_over and self.chess_game.current_player == 'black':
                ai_move = self.ai.get_best_move(self.chess_game)
                if ai_move:
                    self.chess_game.make_move(ai_move)
                    print(f"AI走法: {ai_move}")
            
            # 绘制界面
            self.draw_board()
            self.draw_pieces()
            self.draw_valid_moves()
            self.draw_info()
            
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()
        print("游戏结束！")

def main():
    """主函数：启动人对战"""
    print("="*60)
    print("中国象棋 - 人机对战")
    print("="*60)
    print("你是红方，先手。点击棋子选择，绿色提示为可落子位置。")
    
    # 设置AI难度（搜索深度）
    try:
        depth = int(input("请输入AI难度（1-5，推荐3）: "))
        depth = max(1, min(5, depth))
    except:
        depth = 3
    
    # 启动游戏
    gui = ChessGUI(ai_depth=depth)
    gui.run()

if __name__ == "__main__":
    main()
