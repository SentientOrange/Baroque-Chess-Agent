'''
Eric Eckert - eric95
Derek Wang - d95wang
CSE 415 sp17
Tanimoto

'''
import sys
from queue import PriorityQueue
import time
''' 
Lower case for black, upper case for white
'''

BLACK = 0
WHITE = 1

INIT_TO_CODE = {'p':2, 'P':3, 'c':4, 'C':5, 'l':6, 'L':7, 'i':8, 'I':9,
  'w':10, 'W':11, 'k':12, 'K':13, 'f':14, 'F':15, '-':0}

CODE_TO_INIT = {0:'-',2:'p',3:'P',4:'c',5:'C',6:'l',7:'L',8:'i',9:'I',
  10:'w',11:'W',12:'k',13:'K',14:'f',15:'F'}
  
OPPONENT = ''
SOURCE_STATES = {}

MOVE_SET = [(-1,-1), (-1,0), (-1,1),(0,-1),(0,1),(1,-1), (1,0), (1,1)]
PAWN_MOVE_SET = [(-1,0), (0,-1),(0,1), (1,0)]

PIECE_PRIORITY = {'k':10000, 'f':80, 'c':60, 'i':40, 'w':30, 'l':20, 'p':10} #TODO decide on an initial piece priority, perhaps change it later 

CURRENT_MOVE_START_TIME = 0;
TIME_LIMIT = 0;


def who(piece): return piece % 2

def parse(bs): # bs is board string
  '''Translate a board string into the list of lists representation.'''
  b = [[0,0,0,0,0,0,0,0] for r in range(8)]
  rs9 = bs.split("\n")
  rs8 = rs9[1:] # eliminate the empty first item.
  for iy in range(8):
    rss = rs8[iy].split(' ');
    for jx in range(8):
      b[iy][jx] = INIT_TO_CODE[rss[jx]]
  return b

INITIAL = parse('''
c l i w k i l f
p p p p p p p p
- - - - - - - -
- - - - - - - -
- - - - - - - -
- - - - - - - -
P P P P P P P P
F L I W K I L C
''')

	
class BC_state:
  def __init__(self, old_board=INITIAL, whose_move=WHITE):
    new_board = [r[:] for r in old_board]
    self.board = new_board
    self.whose_move = whose_move;

  def __repr__(self):
    s = ''
    for r in range(8):
      for c in range(8):
        s += CODE_TO_INIT[self.board[r][c]] + " "
      s += "\n"
    if self.whose_move==WHITE: s += "WHITE's move"
    else: s += "BLACK's move"
    s += "\n"
    return s
  
  def __lt__(self, state2):
    return True
    
class state_eval:
  def __init__(self, eval_value, state):
    self.state = state
    # if eval is 0
    if not eval_value:
      eval_value += 0.001
    self.eval_value = 1 / eval_value
  
  def __lt__(self, state2):
    if (self.state.whose_move == WHITE):
      return state2.eval_value > self.eval_value
    
  def __eq__(self, state2):
    return state2.eval_value == self.eval_value
    
  # def __gt__(self, state2):
  #   return state2.eval_value < self.eval_value
    
  def __cmp__(self, other):
    return cmp(self.eval_value, other.eval_value)
    
def test_starting_board():
  init_state = BC_state(INITIAL, WHITE)
  print(init_state.board)
  print(init_state)
  
def introduce():
  intro = "This is the Baroque Chess player, Sentient Orange.\nIt was created by d95wang and eric95 for CSE 415.\n"
  intro += "This orange has been waiting to play chess in a tournament since the day it came off the decision tree."
  return intro
  
def nickname():
  return "SentientOrange"
  
def can_move(board, destRow, destCol):
  if destCol > 7 or destCol < 0:
    return False
  elif destRow > 7 or destRow < 0:
    return False
  return (board[destRow][destCol] == 0)
    
def makeMove(currentState, currentRemark, timeLimit=10000):
  new_remark = "Your move!"
  global CURRENT_MOVE_START_TIME
  global TIME_LIMIT
  CURRENT_MOVE_START_TIME= int(time.time() * 1000)
  TIME_LIMIT = timeLimit
  best_move = currentState
  depth_limit = 0

  while (not time_is_up()):
    best_move = exploreMoves(currentState, 0, depth_limit)
    depth_limit += 1
  if not time_is_up():
    print("hola")
  print(best_move)
  return  [["", best_move], new_remark]



  
# This function takes in a state and explores each potential state in order to determine which move to make
# Use Iterative Deepening Down to level 3 to check three moves ahead
# For each state in the list, see which one has the best value
# First Construct a priority queue maximizing all the potential moves you can make
# Alpha Beta Pruning to lower cost of search
def exploreMoves(currentState, level, depth_limit):
  #If depthlimit was reached, return a static evaluation
  if (level == depth_limit):
    # if not time_is_up():
    #   print("evaled a state")
    return staticEval(currentState)
  else:
    p_queue = PriorityQueue()
    piece_locations = {}
    board = currentState.board
    next_mover = 0 if currentState.whose_move == WHITE else 1
    current_mover = currentState.whose_move
    subtree_summation = 0
    # Get all active piece locations
    #print("Finding all pieces...")
    
    for row in range(8):
      for column in range(8):
        if (board[row][column] != 0):
          
          if board[row][column]%2 == current_mover or board[row][column] == 14 + next_mover:
            temp_piece = CODE_TO_INIT[board[row][column]]
            if temp_piece in piece_locations:
              piece_locations[temp_piece].append((row, column))
            else:
              piece_locations[temp_piece] = [(row, column)]
    
    #we don't even need to look at pieces that are frozen so we can remove them from the dictionary
    #Freezers are represented by 14 and 15, thus adding 14 to next_mover yields the opposing freezer
    opposing_freezer = piece_locations[CODE_TO_INIT[14+next_mover]][0]
    #Tuple coordinates of all frozen pieces
    #print(opposing_freezer)
    #find adjacents from opposing freezer
    frozen = is_adjacent(board, opposing_freezer[0], opposing_freezer[1], next_mover)
    for frozen_piece in frozen:
      #remove frozen pieces from the dictionary
      #print(frozen_piece)
      piece_to_delete = board[frozen_piece[0]][frozen_piece[1]]
      # print(piece_to_delete)
      # print(piece_locations)
      # print(CODE_TO_INIT[piece_to_delete])
      locations_list = piece_locations[CODE_TO_INIT[piece_to_delete]]
      if len(locations_list) > 0:
        locations_list.remove((frozen_piece[0], frozen_piece[1]))
      else:
        del piece_locations[CODE_TO_INIT[piece_to_delete]]
    
    # For each piece on the current state, evaluate the state of the board after you make all possible moves
    #print("looking at each piece")
    for piece_type in piece_locations:
      # Check the piece is movable currently
      #print("piece is being examined")
      if (who(INIT_TO_CODE[piece_type]) == current_mover):
        #print("This piece belongs to the person who is going right now")
        
        #Then iterate through the remaining pieces
        for piece_coord in piece_locations[piece_type]:
          #print("Piece: ", piece_type)
          #print("At: ", piece_coord)
          #print(piece_locations)
          # CHECK ALL POSSIBLE MOVES FOR THIS PIECE in all directions
          move_set = MOVE_SET
          # If it's a pawn, change the moveset to PAWN_MOVE_SET
          if piece_type.lower() == "p":
            move_set = PAWN_MOVE_SET
          for move in move_set:
            multiplier = 1
            #stopped = False
            for i in range(7):
              # Check move in a single direction for a single displacement
              #Kings are represented by 12 and 13, thus adding 12 to current_mover yields the current king
              current_king = piece_locations[CODE_TO_INIT[12+current_mover]][0]
              
              
              #calculate total x and y displacement by multiplying multiplier by move units
              displacement = tuple([multiplier*x for x in move])
              #print("Trying to move to: ")
              #print(piece_coord[0] + displacement[0], ", ", piece_coord[1] + displacement[1])
              temp_board = copy_board(board)
              #move_result = check_move(temp_board, piece_coord[0], piece_coord[1], displacement[0], displacement[1], current_mover, current_king, move)
              move_result = check_move(temp_board, piece_coord[0], piece_coord[1], multiplier, current_mover, current_king, move)
              # The move failed. Stop checking this direction
              if not move_result:
                #print("Cannot move")
                break
              
              #print("Possible move: ")
              #print_board(move_result)
              # Create new state_eval object containing a new state and the eval number
              
              #The queue contains all options at this level organized by evaluation. 
              # priority queue should be used to dermine which subtrees we look at first
              # 1) evaluate current possible move (temp_state)
              #print("Evaluating move...")
              temp_state = BC_state(move_result, next_mover)
              # 2) place in priority queue
              #print("Placing move in queue...")
              p_queue.put((temp_state, exploreMoves(temp_state, level + 1, depth_limit)))
            
              # temp_state = BC_state(move_result, next_mover)
              # #evaluate priority and/or beneficience of move SUBTREE
              # priority = -exploreMoves(temp_state, level + 1, depth_limit) if level % 2 == 1 else exploreMoves(temp_state, level + 1, depth_limit)
              # #The queue contains all options at this level organized by evaluation. 
              # # priority queue should be used to dermine which subtrees we look at first
              # # 1) evaluate current possible move (temp_state)
              # # 2) place in priority queue
              # # 3) run recursion on the whole priority queue
              # p_queue.put(state_eval(temp_state, priority))
              
              #If it's the king, break after a single move
              if (piece_type.lower() == "k"):
                break
        
              multiplier += 1
        # val = 0      
        # while not p_queue.empty():
        #   val += p_queue.get()
          
        # print(val)
          
        # return val 
      # else:
      #   print("this piece does not belong to the current mover")
    
    # this is where you'd want to return the value of the current node, which should
    # be the summation of the priorities of the subtrees, which represents overall how beneficial
    # it is to be at this node
    
    
    #If we are at top level, we only need to pick the move which yields the highest
    #subtree value. We create a dict to store tuples of each state and its subtree value.
    values = []
    
    # 3) for each item in the priority queue, take a look at its subtree
    # print("Calculating subtree values...")
    while not p_queue.empty():
      #take tuple from front of queue
      state_tuple = p_queue.get()
      #print (state_tuple[1])
      # 4) Calculate the subtree's value using that state as a parameter
      # Whether or not the subtree is min or max should be dependent on if its Black or White moving
      
      #subtree_value = -exploreMoves(state_tuple[0], level + 1, depth_limit) if current_mover == WHITE else exploreMoves(state_tuple[0], level + 1, depth_limit)
      subtree_value = exploreMoves(state_tuple[0], level + 1, depth_limit)
      
      # print("Level ", level, " Subtree value: ", subtree_value)
      # If we're at top level, we just need to collect all the subtree values to compare them later
      if level == 0:
        values.append((subtree_value, state_tuple[0]))
        
      # 5) Add it to the summation of subtree values
      else:
        subtree_summation += subtree_value
        
    # Print the time before a state returns
    # show_time()
    # IF we're at top level, return the state with the maximum subtree value
    if level == 0:
      #This lambda iterates through and finds the maximum subtree_value in the list
      #And then returns it's respective state as the Best move.
      #return MAX if current mover is white
      #return MIN if current player is black
      #print("Returning best move...")
      
      if current_mover == WHITE:
        return max(values, key=lambda item:item[0])[1]
      else:
        return min(values, key=lambda item:item[0])[1]
    # 6) return the value up the callstack
    else:
      #print("Returning node value: ",subtree_summation)
      #print("At depth: ", level)
      return subtree_summation
      

def check_move(board, current_row, current_col, multiplier, whose_move, current_king, move):
  
  displacement = tuple([multiplier*x for x in move])
  
  new_row = current_row + displacement[0]
  if new_row > 7 or new_row < 0: return False
  new_col = current_col + displacement[1]
  if new_col > 7 or new_col < 0: return False
  piece = board[current_row][current_col]
  piece_type = CODE_TO_INIT[piece].lower()
  captures = []
  temp_board = board
  imitator_non_standard_capture = False
  
  if piece_type == "-":
    print("trying to move empty space")
    return False
  
  #print(piece_type, " trying to move to: ")
  #print(new_row, ", ", new_col)
  
  # Check if move is legal
  # Coordinators, imitators, pawns, freezers, withdrawers all can capture through standard movement
  can_standard_move = can_move(board, new_row, new_col)
  
  #If the piece cannot more normally, check special cases (king capture, leaper capture, imitator capture as a leaper)
  if not can_standard_move:
    if piece_type == "k":
      captures = check_king_capture(board, new_row, new_col, whose_move)
    
    elif (piece_type == "l" or piece_type == "i") and multiplier == 1:
      #print("checking leaper from: ",current_row,", ",current_col)
      #print("checking leaper capture: ",new_row,", ",new_col)
      if not move in PAWN_MOVE_SET:
        return False
        
      leap_to_row = new_row + move[0]
      leap_to_col = new_col + move[1]
      
      #Check if leaping space is in bounds and empty
      if not in_bounds(leap_to_row,leap_to_col) or board[leap_to_row][leap_to_col]:
        return False
        
      #Capture the piece
      captures = check_leaper_capture(board, new_row, new_col, whose_move)
      #only need to check the first element in the captures list
      # print("Captures: ")
      # print(captures)
      
      #If it is an imitator, check if the capture was a leaper
      if captures and piece_type == "i" and not CODE_TO_INIT[board[captures[0][0]][captures[0][1]]].lower() == "l":
        captures = []
      else:
        imitator_non_standard_capture = True
      
    #If there were no legal normal moves, and then there were captures if it was a king
    #or leaper, then the move failed.
    if not captures:
      #print("Cannot move")
      return False
      
    #However if there were captures by a king or leaper, execute the moves. These
    #are special cases so they need to be executed separately
    elif piece_type == "k":
      temp_board[new_row][new_col] = piece
      temp_board[current_row][current_col] = 0
      #Clear captures because technically we've already replaced the captured piece
      #we don't want to accidently remove the king.
      captures = []
      
    elif piece_type == "l" or piece_type == "i":
      #move leaper to space beyond the capture
      new_row = current_row + 2 * move[0]
      if new_row > 7 or new_row < 0: return False
      new_col = current_col + 2 * move[1]
      if new_col > 7 or new_col < 0: return False
      temp_board[new_row][new_col] = piece
      temp_board[current_row][current_col] = 0
      
      
  else:
    # Execute the move if it can standard move
    #print_board(temp_board)
    temp_board[new_row][new_col] = piece
    temp_board[current_row][current_col] = 0
    #print(piece_type)
    #print("Standard move to: ")
    #print(new_row, ", ", new_col)
    #print_board(temp_board)
    
  #Check standard move captures AND imitator captures (if it already executed a leaper capture)
  if can_standard_move or imitator_non_standard_capture:
    #Check if move has captured anything (this must go in hand with the movement since the capture will result in a move)
    # Coordinators, imitators, pawns, freezers, withdrawers all can capture through standard movement
    #all other pieces will just pass through as a normal move
    # freezer doesn't need a method because it doesn't capture, and the pieces
    #it freezes will be calculated in evaluation function
    
    #Don't include elses because imitators can make multiple captures 
    n = 1
    
    #IF it's an imitator, iterate over captures twice to make sure it makes all legal captures
    #at the same time, possibly all 3.
    if piece_type == "i":
      n = 3
      
    for i in range(n):
      if piece_type == "p" or piece_type == "i":
        
        captures = check_pawn_capture(temp_board, new_row, new_col, whose_move)
        if captures and piece_type == "i":
          #print("pawn captured")
          #print(captures)
        #If it's an imitator

          captures = check_imitator_capture_legal(board, captures, "p")
      ''' IS this supposed to be old row?'''
      if piece_type == "w" or piece_type == "i":
        captures = check_withdrawer_capture(board, current_row, current_col, move, whose_move)
        if piece_type == "i":
          captures = check_imitator_capture_legal(board, captures, "w")
        
      if piece_type == "c" or piece_type == "i":
        captures = check_coordinator_capture(board, new_row, new_col, current_king, whose_move)
        
        if piece_type == "i":
          captures = check_imitator_capture_legal(board, captures, "c")
      #Remove captures
      for capture in captures:
        temp_board[capture[0]][capture[1]] = 0
        
      
  # execute captures
  for capture in captures:
    temp_board[capture[0]][capture[1]] = 0
      
  #Return new board
  return temp_board
    
def check_imitator_capture_legal(board, captures, imitation):
  captured_imitation = False
    #Check all potential captures it could have made
  for capture in captures:
    #Check if it captured a coordinator
    if CODE_TO_INIT[board[capture[0]][capture[1]]].lower() == imitation:
      captured_imitation = True
  #If it didn't capture a coordinator, the move wasn't legal. Clear captures list.
  if not captured_imitation:
    captures = []
      
  return captures
#Check a leaper capture. This capture interprets an attempt to move to an adjacent
#spot as capturing, as opposed to attempting to move to a spot beyond. This is
#easier to implement because the way our movement loop is designed, it keeps 
#extending moves until it finds an illegal move, then stops. If we tried to implement
#the leap beyond then we'd have to keep considering moves even after an illegal one
def check_leaper_capture(board, new_row, new_col, whose_move):
  
  captured = []
   
  if is_opposing_piece(board, new_row, new_col, whose_move):
    captured.append((new_row, new_col))
  
  #print(captured)
  return captured
    
#Check if the withdrawer captured something
def check_withdrawer_capture(board, old_row, old_col, move, whose_move):
  
  captured = []
  
  #The withdrawer capture is essentially one space in the opposite direction of the move
  away_row = old_row - move[0]
  away_col = old_col - move[1]
  
  if is_opposing_piece(board, away_row, away_col, whose_move):
    captured.append((away_row, away_col))
    
  return captured
    
#Check if the king captured something
def check_king_capture(board, new_row, new_col, whose_move):
  
  captured = []
  
  if is_opposing_piece(board, new_row, new_col, whose_move):
    captured.append((new_row, new_col))
  
  return captured
    
#Check if the coordinator captured pieces
def check_coordinator_capture(board, new_row, new_col, current_king, whose_move):
  captured = []
  #Check row of coordinator, col of king
  if is_opposing_piece(board, new_row, current_king[1], whose_move):
    captured.append((new_row, current_king[1]))
  #check col of coordinator, row of king
  if is_opposing_piece(board, current_king[0], new_col, whose_move):
    captured.append((current_king[0], new_col))
    
  return captured
     
#Check if the pawn captured pieces 
def check_pawn_capture(board, new_row, new_col, whose_move):

  captured = []
  #new_adjacents = is_adjacent_pawn(board, new_row, new_col)
  
  # adjacents = []
  for direction in PAWN_MOVE_SET:
    other_row = new_row + direction[0]
    other_col = new_col + direction[1]
    ally_row = new_row + 2 * direction[0]
    if ally_row > 7 or ally_row < 0: continue
    ally_col = new_col + 2 * direction[1]
    if ally_col > 7 or ally_col < 0: continue

    #print("Opponent space: ",other_row,",",other_col)
    #print("Ally space:",ally_row,",",ally_col)
    if is_opposing_piece(board, other_row, other_col, whose_move) and is_ally_piece(board, ally_row, ally_col, whose_move):
      captured.append((other_row, other_col))
          
  return captured
          
  # return adjacents
 
# Given a coordinate for another piece and whose move, determines if piece is an
#opposing piece
def is_opposing_piece(board, row, col, whose_move):
  if in_bounds(row,col):
    other_piece = board[row][col]
    #print("row: ", row, " col: ", col)
    #print(other_piece)
    #print(whose_move)
    return (other_piece % 2 != whose_move) and (other_piece != 0)
    
def is_ally_piece(board, row, col, whose_move):
  if in_bounds(row,col):
    other_piece = board[row][col]
    #print("row: ", row, " col: ", col)
    #print(other_piece)
    #print(whose_move)
    return (other_piece % 2 == whose_move) and (other_piece != 0)
  
def in_bounds(row, col):
  return not (((row > 7) or (row < 0)) or ((col > 7) or (col < 0)))
  
# checks if there are enemy pieces adjacent to the indicated coordinate. 
# If so, returns a list of coordinates where the adjacents are
def is_adjacent(board, row, col, whose_move):
  #Check all directions
  adjacents = []
  for direction in MOVE_SET:
    new_row = row + direction[0]
    #Check if new row is out of bounds
    if new_row > 7 or new_row < 0:
      continue
    new_col = col + direction[1]
    #check if new col is out of bounds
    if new_col > 7 or new_col < 0:
      continue
    #print("row: ", new_row, " col: ", new_col)
    if board[new_row][new_col] != 0:
      #Store coordinate as tuple (row, col)
      #Check if piece is opposing
      if is_opposing_piece(board, new_row, new_col, whose_move):
        adjacents.append((new_row, new_col))
      
  return adjacents
    
# def is_adjacent_pawn(board, row, col):
#   #Check all directions
#   adjacents = []
#   for direction in PAWN_MOVE_SET:
#     new_row = row + direction[0]
#     new_col = col + direction[1]
#     if board[new_row][new_col] != "-":
#       #Store coordinate as tuple (row, col)
#       #Check if piece is opposing
#       if is_opposing_piece(board, row, col, whose_move):
#         adjacents.append((new_row, new_col))
      
#   return adjacents
  
'''
  # MAY USE RECURSION
  # For each value in the priority queue, place examine states using alpha beta pruning
  # You can reuse the queue because it should be empty now
    # Create val objects for each observed state that prioritizes the maximum minimum value
    
  # If you have time you can start looking at the next maximized values
'''    

def prepare(opponent):
  OPPONENT = opponent

# Provides a quick evaluation of whether the state is advantageous for which player
#TODO: are we sure white and black should be hard-coded?
def staticEval(state):
  board = state.board
  white_score = 0
  black_score = 0
  
  # Check the live pieces and see who has more of each piece
  # There is a different weight associated with each piece
  for row in range(8):
    for column in range(8):
      piece = board[row][column]
      if (piece != 0):
        if (who(piece) == WHITE):
          white_score += PIECE_PRIORITY[CODE_TO_INIT[piece].lower()]
        else:
          black_score += PIECE_PRIORITY[CODE_TO_INIT[piece].lower()]
        
  # Return Values for this game
  if white_score > black_score:
    return 1
  elif white_score == black_score:
    return 0
  else:
    return -1
    
def time_is_up():
  return (((int(round(time.time() * 1000)) - CURRENT_MOVE_START_TIME)) > TIME_LIMIT - 4)

def show_time():
  print (((int(round(time.time() * 1000)) - CURRENT_MOVE_START_TIME)))
      
def print_board(board):
  s = ''
  for r in range(8):
    for c in range(8):
      s += CODE_TO_INIT[board[r][c]] + " "
    s += "\n"
  print(s)
  
def copy_board(board):
  return [r[:] for r in board]
# test_starting_board()