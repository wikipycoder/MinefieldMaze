import sys
import random
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout,
                           QPushButton, QLabel, QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import time

class MinefieldEnvironment:
    """Environment class for the Minefield Maze game."""
    
    def __init__(self, size=5, mine_probability=0.3):
        self.size = size
        self.mine_probability = mine_probability
        self.reset()
        
    def reset(self):
        """Reset the environment to initial state."""
        self.grid = np.zeros((self.size, self.size))
        # Place mines randomly
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < self.mine_probability and (i, j) != (0, 0) and (i, j) != (self.size-1, self.size-1):
                    self.grid[i][j] = 1
        
        self.current_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """Convert current position to state number."""
        return self.current_pos[0] * self.size + self.current_pos[1]
    
    def step(self, action):
        """Take action and return new state, reward, and done flag."""
        if self.done:
            return self._get_state(), 0, True
        
        # Action mapping: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = [self.current_pos[0] + moves[action][0],
                  self.current_pos[1] + moves[action][1]]
        
        # Check if move is valid
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size):
            self.current_pos = new_pos
            
            # Check if stepped on mine
            if self.grid[self.current_pos[0]][self.current_pos[1]] == 1:
                self.done = True
                return self._get_state(), -100, True
            
            # Check if reached goal
            if self.current_pos == self.goal_pos:
                self.done = True
                return self._get_state(), 100, True
            
            return self._get_state(), -1, False  # Small negative reward for each step
        
        return self._get_state(), -5, False  # Penalty for invalid move

class QLearningAgent:
    """Q-learning agent for the Minefield Maze game."""
    
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_table = np.zeros((state_size, action_size))
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)
        return np.argmax(self.q_table[state])
    
    def train(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm."""
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MinefieldGame(QMainWindow):
    """Main game window using PyQt5."""
    
    def __init__(self):
        super().__init__()
        self.size = 5
        self.cell_size = 60
        self.env = MinefieldEnvironment(self.size)
        self.agent = QLearningAgent(self.size * self.size, 4)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Minefield Maze')
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create grid
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(2)
        
        # Create grid cells
        self.cells = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                cell = QLabel()
                cell.setFixedSize(self.cell_size, self.cell_size)
                cell.setAlignment(Qt.AlignCenter)
                cell.setStyleSheet('background-color: lightgray; border: 1px solid gray')
                self.grid_layout.addWidget(cell, i, j)
                row.append(cell)
            self.cells.append(row)
        
        layout.addWidget(self.grid_widget)
        
        # Create control buttons
        button_layout = QHBoxLayout()
        self.train_button = QPushButton('Train Agent')
        self.train_button.clicked.connect(self.train_agent)
        self.play_button = QPushButton('Play Game')
        self.play_button.clicked.connect(self.play_game)
        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_game)
        
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)
        
        # Add status label
        self.status_label = QLabel('Welcome to Minefield Maze!')
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.update_grid()
        self.show()
    
    def update_grid(self):
        """Update the grid display."""
        for i in range(self.size):
            for j in range(self.size):
                cell = self.cells[i][j]
                
                # Set cell colors
                if [i, j] == self.env.current_pos:
                    cell.setStyleSheet('background-color: yellow; border: 1px solid gray')
                elif [i, j] == self.env.goal_pos:
                    cell.setStyleSheet('background-color: blue; border: 1px solid gray')
                elif self.env.grid[i][j] == 1:
                    cell.setStyleSheet('background-color: red; border: 1px solid gray')
                else:
                    cell.setStyleSheet('background-color: green; border: 1px solid gray')
    
    def train_agent(self):
        """Train the Q-learning agent."""
        self.status_label.setText('Training agent...')
        episodes = 1000
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.get_action(state)
                next_state, reward, done = self.env.step(action)
                self.agent.train(state, action, reward, next_state)
                state = next_state
            
            if episode % 100 == 0:
                self.status_label.setText(f'Training episode {episode}/{episodes}')
                QApplication.processEvents()
        
        self.status_label.setText('Training complete!')
    
    def play_game(self):
        """Play the game using the trained agent."""
        self.env.reset()
        self.update_grid()
        self.status_label.setText('Playing game...')
        
        def make_move():
            if not self.env.done:
                state = self.env._get_state()
                action = self.agent.get_action(state)
                _, reward, done = self.env.step(action)
                self.update_grid()
                
                if done:
                    if reward > 0:
                        self.status_label.setText('Success! Goal reached!')
                    else:
                        self.status_label.setText('Game Over! Hit a mine!')
                    timer.stop()
        
        timer = QTimer(self)
        timer.timeout.connect(make_move)
        timer.start(500)  # Make a move every 500ms
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.env.reset()
        self.update_grid()
        self.status_label.setText('Game reset!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = MinefieldGame()
    sys.exit(app.exec_())